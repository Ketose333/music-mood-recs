import { chromium } from 'playwright';

const appUrl = process.env.APP_URL;
const appReadyText = process.env.APP_READY_TEXT;
const authUrlPattern = /share\.streamlit\.io\/.*(?:auth|login)|\/auth\/app|\/-\/login(?:[/?]|$)/i;
const wakeButtonPattern = /yes,?\s*get this app back up|wake(?:\s+up)?|get this app back up/i;
const appSelector = '[data-testid="stAppViewContainer"]';

if (!appUrl || !appReadyText) {
  throw new Error('APP_URL과 APP_READY_TEXT 환경 변수가 필요합니다.');
}

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();
let detectedAuthUrl;
let signalAuthDetected;
const authDetected = new Promise((resolve) => {
  signalAuthDetected = resolve;
});

page.on('response', (response) => {
  const location = response.headers().location;
  const authUrl = [response.url(), location].find((url) => url && authUrlPattern.test(url));

  if (authUrl && !detectedAuthUrl) {
    detectedAuthUrl = authUrl;
    signalAuthDetected('auth');
  }
});

function assertPublicApp() {
  if (detectedAuthUrl || authUrlPattern.test(page.url())) {
    throw new Error(
      `Streamlit 인증 페이지로 이동했습니다 (${detectedAuthUrl ?? page.url()}). ` +
        'Community Cloud에서 앱을 Public으로 설정한 후 다시 실행하세요.',
    );
  }
}

try {
  console.log(`Streamlit 앱에 접속합니다: ${appUrl}`);
  const response = await page.goto(appUrl, {
    waitUntil: 'domcontentloaded',
    timeout: 60_000,
  });

  if (!response) {
    throw new Error('앱 접속에 대한 HTTP 응답을 받지 못했습니다.');
  }

  assertPublicApp();

  const wakeButton = page.getByRole('button', { name: wakeButtonPattern }).first();
  const wakeLink = page.getByRole('link', { name: wakeButtonPattern }).first();

  if (await wakeButton.isVisible().catch(() => false)) {
    console.log('sleep 화면을 감지해 앱 깨우기 버튼을 클릭합니다.');
    await wakeButton.click();
  } else if (await wakeLink.isVisible().catch(() => false)) {
    console.log('sleep 화면을 감지해 앱 깨우기 링크를 클릭합니다.');
    await wakeLink.click();
  } else {
    console.log('앱이 이미 깨어 있거나 부팅 중입니다.');
  }

  const loadResult = await Promise.race([
    Promise.all([
      page.locator(appSelector).waitFor({ state: 'visible', timeout: 180_000 }),
      page.getByText(appReadyText, { exact: false }).first().waitFor({
        state: 'visible',
        timeout: 180_000,
      }),
    ]).then(() => 'ready'),
    authDetected,
  ]);

  if (loadResult === 'auth') {
    assertPublicApp();
  }

  assertPublicApp();

  const sleepPromptVisible = await page
    .getByText(/app has gone to sleep|wake (?:it|this app) (?:back )?up/i)
    .first()
    .isVisible()
    .catch(() => false);

  if (sleepPromptVisible) {
    throw new Error('wake 시도 후에도 sleep 화면이 남아 있습니다.');
  }

  console.log(`Streamlit 앱 로딩을 확인했습니다: ${page.url()}`);
} catch (error) {
  console.error(`Keep-alive 실패: ${error.message}`);
  process.exitCode = 1;
} finally {
  await browser.close();
}
