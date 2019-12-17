# Machine Learning for GNSS
This project aims to study usability of ML tools in GNSS positioning. Its core contains ML, GNSS and Utils classes/modules/scripts along with usage examples.

## Useful links
* [IGS (International GNSS Service)](http://www.igs.org/)
* [IGS satellite precise products](http://www.igs.org/products)
* [Access to IGS products](https://kb.igs.org/hc/en-us/articles/115003935351)
* Satellite clock data file formats:
    * ftp://ftp.igs.org/pub/data/format/rinex304.pdf
    * ftp://igs.org/pub/data/format/rinex_clock300.txt
    * ftp://igs.org/pub/data/format/sp3c.txt

## How to run
* ./train_multiple_networks.py data/ Clock_bias 32 20 0.8 nets/ 0.0800576415016
* ./predict_multiple_clocks.py data/ Clock_bias nets/ 32 96 0.0800576415016 predictions/ 2011.0 0.003
* ./compare_lstm_to_others.py predictions/G01_lstm.csv igu_predicted/G01.csv igu_predicted/G01.csv 