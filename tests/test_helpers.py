from unittest import main, TestCase

from disvis.helpers import RestraintParser


class TestRestraintParser(TestCase):

    def test_parse_line(self):
        p = RestraintParser()
        line = 'restraint (1.A@CA or 2.B) (4.C or -8 or 1@CA) 1 -10.4'
        r = p.parse_line(line)
        print r

        # Simple restraint format
        line = 'A 1 CA I -10 BAF 1 10'
        restraint = p.parse_line(line)
        print restraint
        

if __name__ == '__main__':
    main()
