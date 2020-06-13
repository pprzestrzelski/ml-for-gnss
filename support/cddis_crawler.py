from ftplib import FTP
import argparse
import datetime
import os

CDDIS_FTP = 'igs.ensg.ign.fr'
IGU_DIR = 'pub/igs/products/'



def get_valid_directories(file_list, start_idx, end_idx):
    valid_directories = []
    for filename in file_list:
        try:
            i = int(filename)
            if start_idx >= i >= end_idx:
                valid_directories.append(filename)
        except Exception as e:
            pass
    return valid_directories


def crawl_igu_dir(ftp, directory, out_dir):
    print(f'Crawling directory : {directory}')
    ftp.cwd(directory)
    files = ftp.nlst()
    for filename in files:
        if 'sp3' in filename.split('.') and filename[:3] == 'igu':
            out_file_path = os.path.join(out_dir, filename)
            print(f'Downloading {filename}')
            out_file = open(out_file_path, 'wb')
            ftp.retrbinary(f'RETR {filename}', out_file.write)
            out_file.close()
    print('Returning to root dir')
    ftp.cwd('..')


def main(ftp_path, start_idx, end_idx, out_dir):
    try:
        print(f'Connecting to {ftp_path}.')
        ftp = FTP(ftp_path)
        ftp.login()
        print(f'Switching directory to {IGU_DIR}')
        ftp.cwd(IGU_DIR)
        files = ftp.nlst()
        igu_dirs = get_valid_directories(files, start_idx, end_idx)
        print(f'Found {len(igu_dirs)} valid directories.')
        for directory in igu_dirs:
            crawl_igu_dir(ftp, directory, out_dir)
        ftp.quit()
    except Exception as e:
        print(f'Error => {e}')


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
    parser.add_argument('-s', '--start_idx',
                        help='csv with clock bias',
                        type= int,
                        required=True)
    parser.add_argument('-e', '--end_idx',
                        help='csv with clock bias',
                        type=int,
                        required=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args.ftp, args.start_idx, args.end_idx, args.output)
