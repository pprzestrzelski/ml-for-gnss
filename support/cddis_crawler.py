import datetime
import scrapy
from scrapy.http import Request

CDDIS_FTP = 'ftp://cddis.gsfc.nasa.gov/gnss/products/'


class CddisSpider(scrapy.Spider):

    def __init__(self, ftp, user='anonymous', pass=''):
        self.ftp = ftp
        self.meta = {'ftp_user': user, 'ftp_password': pass}
    
    def start_requests(self):
        yield Request(ftp, meta=self.meta)

    def parse(self, response):
        print response.body


def main(ftp, start_date, end_date, out_dir):
    pass

def parse_arguments():
    desc = 'Downloads all clock bias files form CDDIS site for given time period'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-f', '--ftp',
                        help='ftp addres that will be crawled',
                        type=str,
                        default=CDDIS_FTP)
    parser.add_argument('-o', '--output',
                        help='directory to which files will be downloaded',
                        type=str,
                        default='output')
    parser.add_argument('-s', '--start_date',
                        help='csv with clock bias',
                        type= lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'),
                        required=True)
    parser.add_argument('-e', '--end_date',
                        help='csv with clock bias',
                        type= lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'),
                        required=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args.ftp, args.start_date, args.end_date)
