class PorterStemmer:
    def Consonant(self, letter):
        if letter == 'a' or letter == 'e' or letter == 'i' or letter == 'o' or letter == 'u':
            return False
        else:
            return True

    def isConsonant(self, stem, i):
        letter = stem[i]
        if self.Consonant(letter):
            if letter == 'y' and Consonant(stem[i-1]):
                return False
            else:
                return True
        else:
            return False

    def isVowel(self, stem, i):
        return not(isConsonant(stem, i))



    def endsWith(self, stem, letter):
        if stem.endswith(letter):
            return True
        else:
            return False


    def containsVowel(self, stem):
        for i in stem:
            if not self.Consonant(i):
                return True
        return False


    def doubleCons(self, stem):
        if len(stem) >= 2:
            if self.isConsonant(stem, -1) and self.isConsonant(stem, -2):
                return True
            else:
                return False
        else:
            return False

    def Form(self, stem):
        form = []
        formStr = ''
        for i in range(len(stem)):
            if self.isConsonant(stem, i):
                if i != 0:
                    prev = form[-1]
                    if prev != 'C':
                        form.append('C')
                else:
                    form.append('C')
            else:
                if i != 0:
                    prev = form[-1]
                    if prev != 'V':
                        form.append('V')
                else:
                    form.append('V')
        for j in form:
            formStr += j
        return formStr

    def M(self, stem):
        form = self.Form(stem)
        m = form.count('VC')
        return m

    
    def cvc(self, stem):
        if len(stem) >= 3:
            f = -3
            s = -2
            t = -1
            third = stem[t]
            if self.isConsonant(stem, f) and self.isVowel(stem, s) and self.isConsonant(stem, t):
                if third != 'w' and third != 'x' and third != 'y':
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def replace(self, orig, rem, rep):
        result = orig.rfind(rem)
        base = orig[:result]
        replaced = base + rep
        return replaced

    def replaceM0(self, orig, rem, rep):
        result = orig.rfind(rem)
        base = orig[:result]
        if self.M(base) > 0:
            replaced = base + rep
            return replaced
        else:
            return orig

    def replaceM1(self, orig, rem, rep):
        result = orig.rfind(rem)
        base = orig[:result]
        if self.M(base) > 1:
            replaced = base + rep
            return replaced
        else:
            return orig

    def s1a(self, stem):
        if stem.endswith('sses'):
            stem = self.replace(stem, 'sses', 'ss')
        elif stem.endswith('ies'):
            stem = self.replace(stem, 'ies', 'i')
        elif stem.endswith('ss'):
            stem = self.replace(stem, 'ss', 'ss')
        elif stem.endswith('s'):
            stem = self.replace(stem, 's', '')
        else:
            pass
        return stem

    def s1b(self, stem):
        flag = False
        if stem.endswith('eed'):
            result = stem.rfind('eed')
            base = stem[:result]
            if self.M(base) > 0:
                stem = base
                stem += 'ee'
        elif stem.endswith('ed'):
            result = stem.rfind('ed')
            base = stem[:result]
            if self.containsVowel(base):
                stem = base
                flag = True
        elif stem.endswith('ing'):
            result = stem.rfind('ing')
            base = stem[:result]
            if self.containsVowel(base):
                stem = base
                flag = True
        if flag:
            if stem.endswith('at') or stem.endswith('bl') or stem.endswith('iz'):
                stem += 'e'
            elif self.doubleCons(stem) and not self.endsWith(stem, 'l') and not self.endsWith(stem, 's') and not self.endsWith(stem, 'z'):
                stem = stem[:-1]
            elif self.M(stem) == 1 and self.cvc(stem):
                stem += 'e'
            else:
                pass
        else:
            pass
        return stem

    def s1c(self, stem):
        if stem.endswith('y'):
            result = stem.rfind('y')
            base = stem[:result]
            if self.containsVowel(base):
                stem = base
                stem += 'i'
        return stem

    def s2(self, stem):
        if stem.endswith('ational'):
            stem = self.replaceM0(stem, 'ational', 'ate')
        elif stem.endswith('tional'):
            stem = self.replaceM0(stem, 'tional', 'tion')
        elif stem.endswith('enci'):
            stem = self.replaceM0(stem, 'enci', 'ence')
        elif stem.endswith('anci'):
            stem = self.replaceM0(stem, 'anci', 'ance')
        elif stem.endswith('izer'):
            stem = self.replaceM0(stem, 'izer', 'ize')
        elif stem.endswith('abli'):
            stem = self.replaceM0(stem, 'abli', 'able')
        elif stem.endswith('alli'):
            stem = self.replaceM0(stem, 'alli', 'al')
        elif stem.endswith('entli'):
            stem = self.replaceM0(stem, 'entli', 'ent')
        elif stem.endswith('eli'):
            stem = self.replaceM0(stem, 'eli', 'e')
        elif stem.endswith('ousli'):
            stem = self.replaceM0(stem, 'ousli', 'ous')
        elif stem.endswith('ization'):
            stem = self.replaceM0(stem, 'ization', 'ize')
        elif stem.endswith('ation'):
            stem = self.replaceM0(stem, 'ation', 'ate')
        elif stem.endswith('ator'):
            stem = self.replaceM0(stem, 'ator', 'ate')
        elif stem.endswith('alism'):
            stem = self.replaceM0(stem, 'alism', 'al')
        elif stem.endswith('iveness'):
            stem = self.replaceM0(stem, 'iveness', 'ive')
        elif stem.endswith('fulness'):
            stem = self.replaceM0(stem, 'fulness', 'ful')
        elif stem.endswith('ousness'):
            stem = self.replaceM0(stem, 'ousness', 'ous')
        elif stem.endswith('aliti'):
            stem = self.replaceM0(stem, 'aliti', 'al')
        elif stem.endswith('iviti'):
            stem = self.replaceM0(stem, 'iviti', 'ive')
        elif stem.endswith('biliti'):
            stem = self.replaceM0(stem, 'biliti', 'ble')
        return stem

    def s3(self, stem):
        if stem.endswith('icate'):
            stem = self.replaceM0(stem, 'icate', 'ic')
        elif stem.endswith('ative'):
            stem = self.replaceM0(stem, 'ative', '')
        elif stem.endswith('alize'):
            stem = self.replaceM0(stem, 'alize', 'al')
        elif stem.endswith('iciti'):
            stem = self.replaceM0(stem, 'iciti', 'ic')
        elif stem.endswith('ful'):
            stem = self.replaceM0(stem, 'ful', '')
        elif stem.endswith('ness'):
            stem = self.replaceM0(stem, 'ness', '')
        return stem

    def s4(self, stem):
        if stem.endswith('al'):
            stem = self.replaceM1(stem, 'al', '')
        elif stem.endswith('ance'):
            stem = self.replaceM1(stem, 'ance', '')
        elif stem.endswith('ence'):
            stem = self.replaceM1(stem, 'ence', '')
        elif stem.endswith('er'):
            stem = self.replaceM1(stem, 'er', '')
        elif stem.endswith('ic'):
            stem = self.replaceM1(stem, 'ic', '')
        elif stem.endswith('able'):
            stem = self.replaceM1(stem, 'able', '')
        elif stem.endswith('ible'):
            stem = self.replaceM1(stem, 'ible', '')
        elif stem.endswith('ant'):
            stem = self.replaceM1(stem, 'ant', '')
        elif stem.endswith('ement'):
            stem = self.replaceM1(stem, 'ement', '')
        elif stem.endswith('ment'):
            stem = self.replaceM1(stem, 'ment', '')
        elif stem.endswith('ent'):
            stem = self.replaceM1(stem, 'ent', '')
        elif stem.endswith('ou'):
            stem = self.replaceM1(stem, 'ou', '')
        elif stem.endswith('ism'):
            stem = self.replaceM1(stem, 'ism', '')
        elif stem.endswith('ate'):
            stem = self.replaceM1(stem, 'ate', '')
        elif stem.endswith('iti'):
            stem = self.replaceM1(stem, 'iti', '')
        elif stem.endswith('ous'):
            stem = self.replaceM1(stem, 'ous', '')
        elif stem.endswith('ive'):
            stem = self.replaceM1(stem, 'ive', '')
        elif stem.endswith('ize'):
            stem = self.replaceM1(stem, 'ize', '')
        elif stem.endswith('ion'):
            result = stem.rfind('ion')
            base = stem[:result]
            if self.M(base) > 1 and (self.endsWith(base, 's') or self.endsWith(base, 't')):
                stem = base
            stem = self.replaceM1(stem, '', '')
        return stem

    def s5a(self, stem):
        if stem.endswith('e'):
            base = stem[:-1]
            if self.M(base) > 1:
                stem = base
            elif self.M(base) == 1 and not self.cvc(base):
                stem = base
        return stem

    def s5b(self, stem):
        if self.M(stem) > 1 and self.doubleCons(stem) and self.endsWith(stem, 'l'):
            stem = stem[:-1]
        return stem

    def stem(self, stem):
        stem = self.s1a(stem)
        stem = self.s1b(stem)
        stem = self.s1c(stem)
        stem = self.s2(stem)
        stem = self.s3(stem)
        stem = self.s4(stem)
        stem = self.s5a(stem)
        stem = self.s5b(stem)
        return stem


  

"""
ps = PorterStemmer()
  
stems = ["program", "programs", "programmer", "programming", "programmers"]
  
for w in stems:
    print(w, " : ", ps.stem(w))  


    output:
    program  :  program
    programs  :  program
    programmer  :  programm
    programming  :  program
    programmers  :  programm

    """
  