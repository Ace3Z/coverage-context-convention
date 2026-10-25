"""Download Tauber's CC BY-SA 4.0 Beowulf gold and check it before it replaces data/brunetti-length.txt.

The file comes from the master branch of old-english-learner-texts/old-english-texts on GitHub (URL), not from a
pinned commit. It is written to data/ under the repository root, which is found from this file's location.
Usage: python -m oevl.fetch_tauber   (no arguments, run_all.sh calls it when the file is missing or empty)
"""
import sys,tempfile
from pathlib import Path
from urllib.request import urlopen
from oevl.experiment import load_tauber
URL='https://raw.githubusercontent.com/old-english-learner-texts/old-english-texts/master/beowulf/raw-data/brunetti-length.txt'
path=Path(__file__).resolve().parents[2]/'data/brunetti-length.txt'
def main():
 """Download the gold, validate a temporary copy and only then overwrite data/brunetti-length.txt.

 Exits with the usage message if given any argument. The download (90 s socket timeout, not a cap on the whole
 download) is parsed with experiment.load_tauber, which skips blank lines, raises on a row without 18 columns or
 with a marked/bare or word-count mismatch, and prints the token and long-mark counts. It must then give 17,361
 tokens (more than the file's 17,244 rows, as some rows hold two or three words). Only then are the raw bytes
 written to path, replacing any existing file, and "Saved <path>" printed. A failed download or check raises
 before data/ is touched. The size check is an assert, so python -O skips it.
 """
 if len(sys.argv)>1:sys.exit('usage: python -m oevl.fetch_tauber   (downloads data/brunetti-length.txt)')
 with urlopen(URL,timeout=90) as response:data=response.read()
 with tempfile.TemporaryDirectory() as d:  # check a temporary copy, so a download failing a check stays out of data/
  tmp=Path(d)/path.name;tmp.write_bytes(data);rows=load_tauber(tmp)
  assert len(rows)==17361,(len(rows),'Unexpected gold size')
 path.write_bytes(data);print('Saved',path)
if __name__=='__main__':main()
