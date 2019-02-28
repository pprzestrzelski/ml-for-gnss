import examples.example1 as e1
import examples.example2 as e2
import examples.example3 as e3
import examples.example4 as e4
import examples.example5 as e5
import examples.scratchpad as sc

ex = {1: e1.main, 2: e2.main, 3: e3.main, 4: e4.main, 5: e5.main}


def main():
    print('Select example (1-{}):'.format(len(ex)))
    try:
        prompt = input('>')
        selection = int(prompt)
        ex[selection]()
    except ValueError:
        print('Selection must be a number')
    except KeyError:
        print('Choose value from range 1 to {}'.format(len(ex)))


if __name__ == '__main__':
    main()
