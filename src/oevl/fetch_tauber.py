"""Download Tauber's CC BY-SA 4.0 Beowulf gold and check it before it replaces data/brunetti-length.txt."""
import tempfile
from pathlib import Path
from urllib.request import urlopen
from oevl.experiment import load_tauber
URL='https://raw.githubusercontent.com/old-english-learner-texts/old-english-texts/master/beowulf/raw-data/brunetti-length.txt'
path=Path(__file__).resolve().parents[2]/'data/brunetti-length.txt'
def main():
 with urlopen(URL,timeout=90) as response:data=response.read()
 with tempfile.TemporaryDirectory() as d:  # parse a temporary copy, so a changed or truncated download never lands in data/
  tmp=Path(d)/path.name;tmp.write_bytes(data);rows=load_tauber(tmp)
  assert len(rows)==17361,(len(rows),'Unexpected gold size')
 path.write_bytes(data);print('Saved',path)
if __name__=='__main__':main()
