# cozychat
Simple tool for analyzing VK or Telegram chat contents.

VK chat's txt can be saved via Kate Mobile (https://vk.com/kate_mobile) app.

Telegram group chat's json can be exported via Telegram Desktop (https://desktop.telegram.org) app.

Python 3.8.1 or higher is required, 3.11 is tested.

Quick start:
```bash
git clone https://github.com/lysukhin/cozychat.git
cd cozychat
pip install -r requirements.txt
python -m dostoevsky download fasttext-social-network-model

cd examples
jupyter-lab --ServerApp.iopub_data_rate_limit=1.0e10
```
Place your txt or json into `examples`, open `2020.ipynb`, change chat filename, chat type and year and run the code.

If you want to export as web-rendered pdf (webpdf):
```
pip install nbconvert[webpdf]
playwright install chromium
```

Links to notebooks on nbviewer:
* [2020](https://nbviewer.jupyter.org/github/lysukhin/cozychat/blob/master/examples/2020.ipynb)
* [2019](https://nbviewer.jupyter.org/github/lysukhin/cozychat/blob/master/examples/2019.ipynb)
* [2018](https://nbviewer.jupyter.org/github/lysukhin/cozychat/blob/master/examples/2018.ipynb)
* [2017](https://nbviewer.jupyter.org/github/lysukhin/cozychat/blob/master/examples/2017.ipynb)
* [2016](https://nbviewer.jupyter.org/github/lysukhin/cozychat/blob/master/examples/2016.ipynb)
* [2015](https://nbviewer.jupyter.org/github/lysukhin/cozychat/blob/master/examples/2015.ipynb)
* [2014](https://nbviewer.jupyter.org/github/lysukhin/cozychat/blob/master/examples/2014.ipynb)
