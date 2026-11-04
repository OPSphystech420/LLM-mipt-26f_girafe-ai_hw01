"""Download the textbook PDF."""
import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

URL = "https://new.mipt.ru/upload/mathematics/study/uchebniki/%D0%9B_%D0%A2%D0%A4%D0%9A%D0%9F_%D0%93%D0%BE%D1%80%D1%8F%D0%B9%D0%BD%D0%BE%D0%B2_%D0%9F%D0%BE%D0%BB%D0%BE%D0%B2%D0%B8%D0%BD%D0%BA%D0%B8%D0%BD_2025.pdf"
SHA256 = "629f4b668ff43abf2a6e621aefd0f50e165ce8e88c52c05769149e1267d033ec"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=".local/textbook.pdf")
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        data = path.read_bytes()
    else:
        with urlopen(URL, timeout=120) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("PDF differs from the expected source")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    print(path)


if __name__ == "__main__":
    main()
