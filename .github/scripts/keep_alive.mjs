import { chromium } from 'playwright';

const appUrl = process.env.APP_URL;
const appReadyText = process.env.APP_READY_TEXT;
const wakeButtonPattern = /(?:yes,?\s*)?(?:get this app back up|wake (?:this app )?up)/i;
const loadTimeoutMs = 180_000;

if (!appUrl || !appReadyText) {
  throw new Error('APP_URL과 APP_READY_TEXT 환경 변수가 필요합니다.');
}

function safeUrl(url) {
  const parsedUrl = url instanceof URL ? url : new URL(url);
  return `${parsedUrl.origin}${parsedUrl.pathname}`;
}

function assertPublicApp(page) {
  const currentUrl = new URL(page.url());
  const isAuthPage =
    (currentUrl.hostname === 'share.streamlit.io' &&
      currentUrl.pathname.includes('/auth/')) ||
    currentUrl.pathname.startsWith('/-/login');

  if (isAuthPage) {
    throw new Error(
      `Streamlit 인증 페이지로 이동했습니다: ${safeUrl(currentUrl)}\n` +
        'keep-alive를 실행하려면 Streamlit Community Cloud에서 앱을 Public으로 설정해야 합니다.',
    );
  }
}

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();

try {
  console.log(`Streamlit 앱에 접속합니다: ${safeUrl(appUrl)}`);
  await page.goto(appUrl, {
    waitUntil: 'domcontentloaded',
    timeout: 60_000,
  });

  const appFrame = page.frameLocator('iframe[title="streamlitApp"]');
  const topLevelContainer = page.locator('[data-testid="stAppViewContainer"]').first();
  const topLevelReadyText = page.getByText(appReadyText, { exact: false }).first();
  const iframeReadyText = appFrame.getByText(appReadyText, { exact: false }).first();
  const wakeButton = page.getByRole('button', { name: wakeButtonPattern }).first();
  const deadline = Date.now() + loadTimeoutMs;
  let wakeClicked = false;
  let readyMode = null;

  async function detectReadyMode() {
    const [hasTopLevelContainer, hasTopLevelReadyText] = await Promise.all([
      topLevelContainer.isVisible().catch(() => false),
      topLevelReadyText.isVisible().catch(() => false),
    ]);

    if (hasTopLevelContainer && hasTopLevelReadyText) {
      return 'top-level';
    }

    if (await iframeReadyText.isVisible().catch(() => false)) {
      return 'iframe';
    }

    return null;
  }

  while (Date.now() < deadline) {
    assertPublicApp(page);
    readyMode = await detectReadyMode();

    if (readyMode) {
      console.log(`앱 정상 로딩을 확인했습니다(${readyMode}): ${safeUrl(page.url())}`);
      break;
    }

    if (!wakeClicked && (await wakeButton.isVisible().catch(() => false))) {
      console.log('sleep 화면을 감지해 wake 버튼을 클릭합니다.');
      await wakeButton.click();
      wakeClicked = true;
    }

    await page.waitForTimeout(2_000);
  }

  assertPublicApp(page);

  if (!readyMode) {
    throw new Error(
      `${loadTimeoutMs / 1_000}초 안에 앱 정상 로딩을 확인하지 못했습니다. 최종 URL: ${safeUrl(page.url())}`,
    );
  }
} catch (error) {
  console.error(`Keep-alive 실패: ${error.message}`);
  process.exitCode = 1;
} finally {
  await browser.close();
}
