import examples.example1 as e1
import examples.example2 as e2
import examples.example3 as e3
import examples.example4 as e4
import examples.example5 as e5
import examples.example6 as e6
import examples.example7 as e7
import examples.example8 as e8
import examples.scratchpad as sc

ex = {1: e1.main, 2: e2.main, 3: e3.main, 4: e4.main, 5: e5.main,
      6: e6.main, 7: e7.main, 8: e8.main}


def main():
    print('Select example (1-{}):'.format(len(ex)))
    prompt = input('>')
    selection = int(prompt)
    ex[selection]()


if __name__ == '__main__':
    main()
