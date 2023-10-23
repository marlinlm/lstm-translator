import xml.etree.ElementTree as ET

class tmxHandler():

    def __init__(self):
        self.tag=None
        self.lang=None
        self.corpus={}

    def handleStartTu(self, tag):
        self.tag=tag
        self.lang=None
        self.corpus={}

    def handleStartTuv(self, tag, attributes):
        if self.tag == 'tu':
            if attributes['{http://www.w3.org/XML/1998/namespace}lang']:
                self.lang=attributes['{http://www.w3.org/XML/1998/namespace}lang']
            else:
                raise Exception('tuv element must has a xml:lang attribute')
            self.tag = tag
        else:
            raise Exception('tuv element must go under tu, not ' + tag)
    
    def handleStartSeg(self, tag, elem):
        if self.tag == 'tuv':
            self.tag = tag
            if self.lang:
                if elem.text:
                    self.corpus[self.lang]=elem.text
            else:
                raise Exception('lang must not be none')
        else:
            raise Exception('seg element must go under tuv, not ' + tag)

    def startElement(self, tag, attributes, elem):
        if tag== 'tu':
            self.handleStartTu(tag)
        elif tag == 'tuv':
            self.handleStartTuv(tag, attributes)
        elif tag == 'seg':
            self.handleStartSeg(tag, elem)
        else:
            pass



    def endElem(self, tag):
        if self.tag and self.tag != tag:
            raise Exception(self.tag + ' could not end with ' + tag)

        if tag == 'tu':
            self.tag=None
            self.lang=None
            self.corpus={}
        elif tag == 'tuv':
            self.tag='tu'
            self.lang=None
        elif tag == 'seg':
            self.tag='tuv'
        # else:
            # pass
            

    def parse(self, filename):
        for event, elem in ET.iterparse(filename, events=('start','end')):
            if event == 'start':
                self.startElement(elem.tag, elem.attrib, elem)
            elif event == 'end':
                if elem.tag=='tu':
                    yield self.corpus
                self.endElem(elem.tag)






