import { chromium } from 'playwright';

const appUrl = process.env.APP_URL;
const appReadyText = process.env.APP_READY_TEXT;
const wakeControlPattern = /^(?:yes,?\s*)?(?:get this app back up|wake (?:this app )?up)[.!]?$/i;
const loadTimeoutMs = 180_000;

if (!appUrl || !appReadyText) {
  throw new Error('APP_URL과 APP_READY_TEXT 환경 변수가 필요합니다.');
}

const targetUrl = new URL(appUrl);
const targetHostname = targetUrl.hostname.toLowerCase();

function safeUrl(url) {
  const parsedUrl = url instanceof URL ? url : new URL(url);
  return `${parsedUrl.origin}${parsedUrl.pathname}`;
}

function safeErrorMessage(error) {
  const message = error instanceof Error ? error.message : String(error);

  return message.replace(/https?:\/\/[^\s"'<>]+/g, (url) => {
    try {
      return safeUrl(url);
    } catch {
      return '[URL 숨김]';
    }
  });
}

function isAuthUrl(url) {
  return (
    (url.hostname.toLowerCase() === 'share.streamlit.io' &&
      url.pathname.includes('/auth/')) ||
    (url.hostname.toLowerCase() === targetHostname &&
      url.pathname.startsWith('/-/login'))
  );
}

function assertFinalUrl(page) {
  const currentUrl = new URL(page.url());

  if (isAuthUrl(currentUrl)) {
    throw new Error(
      `Streamlit 인증 페이지로 이동했습니다: ${safeUrl(currentUrl)}\n` +
        'keep-alive를 실행하려면 Streamlit Community Cloud에서 앱을 Public으로 설정해야 합니다.',
    );
  }

  if (currentUrl.hostname.toLowerCase() !== targetHostname) {
    throw new Error(`대상 앱이 아닌 호스트로 이동했습니다: ${safeUrl(currentUrl)}`);
  }
}

async function waitForPublicAppBootstrap(page) {
  const deadline = Date.now() + 15_000;

  while (Date.now() < deadline) {
    const currentUrl = new URL(page.url());

    if (!isAuthUrl(currentUrl)) {
      assertFinalUrl(page);
      return;
    }

    await page.waitForTimeout(500);
  }

  assertFinalUrl(page);
}

let browser;

try {
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();

  console.log(`Streamlit 앱에 접속합니다: ${safeUrl(targetUrl)}`);
  await page.goto(appUrl, {
    waitUntil: 'domcontentloaded',
    timeout: 60_000,
  });
  await waitForPublicAppBootstrap(page);

  const appFrame = page.frameLocator('iframe[title="streamlitApp"]');
  const topLevelContainer = page.locator('[data-testid="stAppViewContainer"]').first();
  const topLevelReadyText = page.getByText(appReadyText, { exact: false }).first();
  const iframeReadyText = appFrame.getByText(appReadyText, { exact: false }).first();
  const deadline = Date.now() + loadTimeoutMs;
  let wakeClicked = false;
  let readyMode = null;

  async function detectReadyMode() {
    const currentHostname = new URL(page.url()).hostname.toLowerCase();

    if (currentHostname !== targetHostname) {
      return null;
    }

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

  async function findWakeControl() {
    for (const role of ['button', 'link']) {
      const control = page.getByRole(role, { name: wakeControlPattern }).first();

      if (await control.isVisible().catch(() => false)) {
        return control;
      }
    }

    return null;
  }

  while (Date.now() < deadline) {
    assertFinalUrl(page);
    readyMode = await detectReadyMode();

    if (readyMode) {
      console.log(`앱 정상 로딩을 확인했습니다(${readyMode}): ${safeUrl(page.url())}`);
      break;
    }

    if (!wakeClicked) {
      const wakeControl = await findWakeControl();

      if (wakeControl) {
        wakeClicked = true;
        console.log('sleep 화면을 감지해 wake 컨트롤을 클릭합니다.');
        assertFinalUrl(page);
        await wakeControl.click();
      }
    }

    await page.waitForTimeout(2_000);
  }

  assertFinalUrl(page);

  if (!readyMode) {
    throw new Error(
      `${loadTimeoutMs / 1_000}초 안에 앱 정상 로딩을 확인하지 못했습니다. 최종 URL: ${safeUrl(page.url())}`,
    );
  }
} catch (error) {
  console.error(`Keep-alive 실패: ${safeErrorMessage(error)}`);
  process.exitCode = 1;
} finally {
  await browser?.close().catch(() => {});
}
