"""Download Tauber's CC BY-SA 4.0 Beowulf gold; verify strict schema before use."""
from pathlib import Path
from urllib.request import urlopen
from experiment import load_tauber
URL='https://raw.githubusercontent.com/old-english-learner-texts/old-english-texts/master/beowulf/raw-data/brunetti-length.txt'
path=Path(__file__).resolve().parents[1]/'data/brunetti-length.txt'
with urlopen(URL,timeout=90) as response:data=response.read()
path.write_bytes(data)
rows=load_tauber(path)
assert len(rows)==17361,(len(rows),'Unexpected gold size')
print('Saved',path)
