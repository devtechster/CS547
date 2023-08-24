
__version__ = "20210123"
COURSE = "cs547"

class Student(object):
    """
    This object will be used to represent a student's identifying information
    in all programming assignments.  Any assignments without an instantiated
    student object of type Student will not be graded.
    """

    def __init__(self, name, anum, email, collabs=None, honor=False):
        # collabs = [  (name, contribution) , ... ]
        self._name = name
        self._anum = anum
        self._email = email
        self._collabs = collabs or []
        self._honor = honor


    def __repr__(self):
        return """
%s %s (uin:%d honor:%s)
  - %s""" % (
            COURSE,
            self._name,
            self._anum,
            self._honor,
            "\n  - ".join( "%s -- %s" % (x,y) for x,y in self._collabs )
            )

