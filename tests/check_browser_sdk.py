"""Exercise the distributed browser assets and worker using known model weights."""
import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detector", type=Path)
    parser.add_argument("embedder", type=Path)
    parser.add_argument("--chromium")
    args = parser.parse_args()
    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "wasm").symlink_to(ROOT / "wasm", target_is_directory=True)
        (root / "frame").symlink_to(ROOT / "tests/test_face_160.raw")
        (root / "det.bin").symlink_to(args.detector.resolve(strict=True))
        (root / "emb.bin").symlink_to(args.embedder.resolve(strict=True))
        scripts = ("detect", "facex", "align", "cache", "simd_check", "facex-sdk")
        (root / "index.html").write_text("".join(f'<script src="/wasm/{s}.js"></script>' for s in scripts))
        handler = functools.partial(SimpleHTTPRequestHandler, directory=directory)
        with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(executable_path=args.chromium)
                    page = browser.new_page()
                    page.goto(f"http://127.0.0.1:{server.server_port}")
                    print(page.evaluate((ROOT / "tests/browser_sdk_smoke.js").read_text()))
                    browser.close()
            finally:
                server.shutdown()
                thread.join()


if __name__ == "__main__":
    main()
