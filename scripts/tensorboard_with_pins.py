"""Run TensorBoard with a higher frontend global-pin limit.

TensorBoard currently hard-codes the global pin limit in its bundled frontend.
This entry point patches that bundle in memory, so the installed TensorBoard
package remains untouched and normal TensorBoard command-line flags continue to
work unchanged.
"""

import io
import sys
import zipfile

from absl import app
from tensorboard import assets
from tensorboard import default
from tensorboard import main_lib
from tensorboard import program
from tensorboard.plugins import base_plugin


MAX_PIN_COUNT = 100
_PIN_LIMIT_TOKEN = "MAX_PIN_COUNT:10"


def patch_index_js(index_js, max_pin_count=MAX_PIN_COUNT):
    """Replace TensorBoard's bundled global pin limit in index.js."""
    replacement = "MAX_PIN_COUNT:%d" % max_pin_count
    if _PIN_LIMIT_TOKEN not in index_js:
        raise RuntimeError(
            "TensorBoard frontend no longer contains the expected pin limit "
            "token; refusing to start with an unverified asset patch."
        )
    return index_js.replace(_PIN_LIMIT_TOKEN, replacement, 1)


def _patched_assets_zip():
    """Return patched web assets as an in-memory ZIP archive."""
    provider = assets.get_default_assets_zip_provider()
    if provider is None:
        raise RuntimeError("TensorBoard webfiles.zip was not found.")

    output = io.BytesIO()
    with provider() as source, zipfile.ZipFile(source) as source_zip:
        with zipfile.ZipFile(output, mode="w") as output_zip:
            for entry in source_zip.infolist():
                content = source_zip.read(entry.filename)
                if entry.filename == "index.js":
                    content = patch_index_js(
                        content.decode("utf-8")
                    ).encode("utf-8")
                output_zip.writestr(entry, content)
    return output.getvalue()


def _assets_zip_provider():
    """Provide a fresh file-like ZIP object for each TensorBoard request."""
    patched_assets = _patched_assets_zip()
    return lambda: io.BytesIO(patched_assets)


def run_main():
    """Initialize and run TensorBoard with patched frontend assets."""
    main_lib.global_init()
    tensorboard = program.TensorBoard(
        plugins=default.get_plugins(),
        assets_zip_provider=_assets_zip_provider(),
    )
    try:
        app.run(tensorboard.main, flags_parser=tensorboard.configure)
    except base_plugin.FlagsError as error:
        print("Error: %s" % error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run_main())
