from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def transcode_av1_to_h264(input_path: Path, output_path: Path) -> Path:
	"""Transcode an AV1 video file to H.264 (yuv420p) using ffmpeg."""
	ffmpeg = shutil.which("ffmpeg")
	if ffmpeg is None:
		raise RuntimeError("ffmpeg was not found on PATH.")

	if not input_path.is_file():
		raise FileNotFoundError(f"Input video not found: {input_path}")

	command = [
		ffmpeg,
		"-y",
		"-hwaccel",
		"none",
		"-i",
		str(input_path),
		"-c:v",
		"libx264",
		"-pix_fmt",
		"yuv420p",
		"-an",
		str(output_path),
	]

	result = subprocess.run(command, capture_output=True, text=True)
	if result.returncode != 0:
		raise RuntimeError(f"ffmpeg transcode failed:\n{result.stderr[-4000:]}")

	if not output_path.is_file() or output_path.stat().st_size == 0:
		raise RuntimeError(f"Output file was not created correctly: {output_path}")

	return output_path


def main() -> None:
	default_input = Path("orange1.mp4").resolve()
	default_output = default_input.with_name("orange1_h264.mp4")

	input_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else default_input
	output_path = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else default_output

	print(f"Input: {input_path}")
	print(f"Output: {output_path}")
	written_path = transcode_av1_to_h264(input_path, output_path)
	print(f"Done: {written_path}")


if __name__ == "__main__":
	main()
