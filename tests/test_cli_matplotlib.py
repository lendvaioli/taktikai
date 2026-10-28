import os
import subprocess
import sys
import textwrap

import pytest

# The script imports mplsoccer (and so matplotlib) only after the CLI entry point has run, as the
# pitch stage does; with the inherited Jupyter backend this used to fail at "import matplotlib".
SCRIPT = textwrap.dedent(
    """
    import sys
    from match_tagger.cli import main

    assert main(["run", "--stage", "unknown", "--video", sys.argv[1]]) == 2
    from match_tagger.pitch.geometry import PitchDims
    from match_tagger.pitch.minimap import pitch_canvas

    image, _ = pitch_canvas(PitchDims(), 100)
    import matplotlib

    print(matplotlib.get_backend().lower(), image.shape[0])
    """
)


def test_cli_survives_jupyter_matplotlib_backend(tmp_path):
    pytest.importorskip("mplsoccer")
    video = tmp_path / "x.mp4"
    video.write_bytes(b"")
    env = {"MPLBACKEND": "module://matplotlib_inline.backend_inline"}
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(video)],
        capture_output=True,
        text=True,
        env={**os.environ, **env},
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.split()[0] == "agg"
