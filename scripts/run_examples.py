import examples.example1 as e1
import examples.example2 as e2
import examples.example3 as e3

ex = {1: e1.main, 2: e2.main, 3: e3.main}


def main():
    print('Select example (1-3):')
    try:
        prompt = input('>')
        selection = int(prompt)
        ex[selection]()
    except ValueError:
        print('Selection must be a number')
    except KeyError:
        print('Choose value from range 1 to 3')


if __name__ == '__main__':
    main()