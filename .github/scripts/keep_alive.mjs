import { chromium } from 'playwright';

const appUrl = process.env.APP_URL;
const appReadyText = process.env.APP_READY_TEXT;
const authUrlPattern = /share\.streamlit\.io\/.*(?:auth|login)|\/auth\/app|\/-\/login(?:[/?]|$)/i;
const wakeButtonPattern = /^\s*(?:yes,\s*)?get this app back up!?\s*$/i;
const appSelector = '[data-testid="stAppViewContainer"]';
const loadTimeoutMs = 180_000;

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
    const rawAuthUrl = detectedAuthUrl ?? page.url();
    let safeAuthUrl = 'Streamlit 인증 경로';

    try {
      const parsedAuthUrl = new URL(rawAuthUrl, page.url());
      safeAuthUrl = `${parsedAuthUrl.origin}${parsedAuthUrl.pathname}`;
    } catch {
      // 로그에 query나 payload가 노출되지 않도록 원본 URL은 출력하지 않는다.
    }

    throw new Error(
      `Streamlit 인증 페이지로 이동했습니다 (${safeAuthUrl}). ` +
        'Community Cloud에서 앱을 Public으로 설정한 후 다시 실행하세요.',
    );
  }
}

function safeUrl(url) {
  try {
    const parsedUrl = new URL(url);
    return `${parsedUrl.origin}${parsedUrl.pathname}`;
  } catch {
    return 'Streamlit 앱 경로';
  }
}

try {
  console.log(`Streamlit 앱에 접속합니다: ${safeUrl(appUrl)}`);
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
  const appContainer = page.locator(appSelector);
  const readyText = page.getByText(appReadyText, { exact: false }).first();
  const deadline = Date.now() + loadTimeoutMs;
  let wakeClicked = false;
  let appReady = false;

  while (Date.now() < deadline) {
    assertPublicApp();

    const appVisible = await appContainer.isVisible().catch(() => false);
    const readyTextVisible = await readyText.isVisible().catch(() => false);

    if (appVisible && readyTextVisible) {
      appReady = true;
      break;
    }

    if (!wakeClicked) {
      let wakeControl;
      let wakeControlName;

      if (await wakeButton.isVisible().catch(() => false)) {
        wakeControl = wakeButton;
        wakeControlName = '버튼';
      } else if (await wakeLink.isVisible().catch(() => false)) {
        wakeControl = wakeLink;
        wakeControlName = '링크';
      }

      if (wakeControl) {
        wakeClicked = true;
        console.log(`sleep 화면을 감지해 앱 깨우기 ${wakeControlName}를 클릭합니다.`);
        await wakeControl.click();
      }
    }

    const remainingMs = deadline - Date.now();
    if (remainingMs <= 0) {
      break;
    }

    const loopResult = await Promise.race([
      authDetected,
      page.waitForTimeout(Math.min(2_000, remainingMs)).then(() => 'tick'),
    ]);

    if (loopResult === 'auth') {
      assertPublicApp();
    }
  }

  assertPublicApp();

  if (!appReady) {
    throw new Error(`${loadTimeoutMs / 1_000}초 안에 앱 정상 로딩을 확인하지 못했습니다.`);
  }

  const sleepPromptVisible = await page
    .getByText(/app has gone to sleep|wake (?:it|this app) (?:back )?up/i)
    .first()
    .isVisible()
    .catch(() => false);

  if (sleepPromptVisible) {
    throw new Error('wake 시도 후에도 sleep 화면이 남아 있습니다.');
  }

  console.log(`Streamlit 앱 로딩을 확인했습니다: ${safeUrl(page.url())}`);
} catch (error) {
  console.error(`Keep-alive 실패: ${error.message}`);
  process.exitCode = 1;
} finally {
  await browser.close();
}
