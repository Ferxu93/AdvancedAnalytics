''' Introduction to classes '''
class Virus:
    possibles = ('Coronavirus', 'Hantavirus', 'Flu') # tuple (immutable object)

    def __init__(self, name, country): # magic functions
        self.name = name
        self.country = country
        self.city = []

    def __repr__(self): # __repr__ => means: representation (magic functions = built-in functions)
        return f'The class contains the virus name: {self.name} and the country: {self.country}, ' \
               f'and the infected cities are: {self.city}'

    def add_city(self, ciudad):
        self.city.append(ciudad)

    @classmethod
    def virus_name(cls, name):
        if name in cls.possibles:
            return name

        else:
            print(f'This virus does not exists in the database: {cls.possibles}')
            print('Try one of does or get out!')

    @staticmethod # dump function ~ lambda (but inside a class) <AS BACKLOG>
    def symptoms():
        print('Very bad symptoms')


virus = Virus('Coronavirus', 'Spain')
virus.virus_name('Hantavirus')
virus.add_city('Madrid')
print(virus)
virus.symptoms()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


