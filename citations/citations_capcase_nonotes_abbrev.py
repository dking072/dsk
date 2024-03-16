#!/usr/bin/env python3
import sys
import ast
from titlecase import titlecase

#Usage: ./script.py bibfile? lol? lol

abbreviations = {
    "Chemical Reviews":"Chem. Rev.",
    "Journal of the American Chemical Society":"J. Am. Chem. Soc.",
    "Journal of Computational Chemistry":"J. Comput. Chem.",
    "Journal of Chemical Theory and Computation":"J. Chem. Theory Comput.",
    "Journal of Chemical Physics":"J. Chem. Phys.",
    "Journal of Physical Chemistry": "J. Phys. Chem.",
    "The Journal of Physical Chemistry":"J. Phys. Chem.", #Apparently the title was changed??? Eh just gonna keep it? huh? lol
    "The Journal of Physical Chemistry A":"J. Phys. Chem. A",
    "The Journal of Physical Chemistry B":"J. Phys. Chem. B",
    "The Journal of Physical Chemistry C":"J. Phys. Chem. C",
    "The Journal of Physical Chemistry Letters":"J. Phys. Chem. Lett.",
    "Nature Communications":"Nat. Commun.",
    "Nature Catalysis":"Nat. Catal.",
    "International Journal of Quantum Chemistry":"Int. J. Quantum Chem.",
    "Surface Science":"Surf. Sci.",
    "Physical Review Letters":"Phys. Rev. Lett.",
    "Physical Review A":"Phys. Rev. A",
    "Physical Review B":"Phys. Rev. B",
    "The Journal of Chemical Physics":"J. Chem. Phys.",
    "Chemical Science":"Chem. Sci.",
    "Reviews of Modern Physics":"Rev. Mod. Phys.",
    "WIREs Computational Molecular Science":"Wiley Interdiscip. Rev. Comput. Mol. Sci.",
    "CHIMIA International Journal for Chemistry":"Chimia (Aarau)", #???? idk dude I guess this is it??? lol
    "Chemical Physics Letters":"Chem. Phys. Lett.",
    "Theoretical Chemistry Accounts":"Theor. Chem. Acc.",
    "Angewandte Chemie International Edition":"Angew. Chem. Int. Ed.",
    "ACS Central Science":"ACS Cent. Sci.",
    "ACS Catalysis": "ACS Catal.",
    "Journal of Catalysis":"J. Catal.",
    "Chemistry of Materials":"Chem. Mater.",
    "Physical Chemistry Chemical Physics":"Phys. Chem. Chem. Phys",
    "Coordination Chemistry Reviews":"Coord. Chem. Rev.",
    "Inorganic Chemistry": "Inorg. Chem.",
    "Chemical Physics": "Chem. Phys.",
    "Computer Physics Communications":"Comput. Phys. Commun.",
    "Theoretica Chimica Acta":"Theor. Chim. Acta.",
    "Molecular Physics":"Mol. Phys.",
    "Trends in Chemistry":"Trends Chem.",
    "Scientific Reports":"Sci. Rep.",
    "Physical Review Materials":"Phys. Rev. Mater.",
    "Machine Learning: Science and Technology":"Mach. Learn.: Sci. Technol.",
    "NPJ Computational Materials":"Npj Comput. Mater.",
    "Accounts of Chemical Research":"Acc. Chem. Res.",
    "Journal of Chemical Information and Modeling":"J. Chem. Inf. Model.",
    "Reviews in Computational Chemistry":"Rev. Comput. Chem.",
    "Pattern Recognition Letters":"Pattern Recognit. Lett.",
    "Chemical Society Reviews":"Chem. Soc. Rev.",
    "International Reviews in Physical Chemistry":"Int. Rev. Phys. Chem.",
    "Wiley Interdisciplinary Reviews: Computational Molecular Science":"Wiley Interdiscip. Rev. Comput. Mol. Sci.",
    "International Journal of Quantum Chemistry":"Int. J. Quantum Chem.",
    "Annual Review of Physical Chemistry":"Annu. Rev. Phys. Chem.",
    "Zeitschrift für Physikalische Chemie":"Z. Phys. Chem.",
    "Annals of Physics":"Ann. Phys. (N. Y.)",
    "The European Physical Journal D":"Eur. Phys. J. D",
    "The Journal of Organic Chemistry":"J. Org. Chem.",
    "Chemical Communications":"Chem. Commun.",
    "Journal of Biological Chemistry":"J. Biol. Chem.",
    "Angewandte Chemie":"Angew. Chem.",
    "Journal of Chemical Education":"J. Chem. Educ.",
    "Nature":"Nature",
    "Nature Materials":"Nat. Mater.",
    "Science Advances":"Sci. Adv.",
    "Advanced Functional Materials":"Adv. Funct. Mater.",
    "Nature Electronics":"Nat. Electron.",
    "Nature Reviews Materials":"Nat. Rev. Mater.",
    "Science":"Science",
    "Journal of Chemical \& Engineering Data":"J. Chem. Eng. Data",
    "ACS Energy Letters":"ACS Energy Lett.",
    "Israel Journal of Chemistry":"Isr. J. Chem.",
    "International Journal of Hydrogen Energy":"Int. J. Hydrogen Energy",
}

journals = list(abbreviations.keys())

for bibfile_name in sys.argv[1:]:

    #Read in lines:
    with open(bibfile_name,"r") as file:
        lines = file.readlines()

    new_lines = []

    def get_inb(line):
        firstbidx = line.find("{")
        lastbidx = len(line) - line[::-1].find("}")
        if firstbidx < 0 or lastbidx < 0:
            print("Error!")
            exit(1)
        return line[firstbidx+1:lastbidx-1]

    def replace_inb(line,replacement):
        firstbidx = line.find("{")
        lastbidx = len(line) - line[::-1].find("}")
        return line[:firstbidx+1] + replacement + line[lastbidx-1:]

    start=False
    startskip=False
    for line in lines:
        if "@" in line:
            start = True
            new_lines += [line]
            continue
        if line == "}\n":
            start = False
            new_lines += [line]
            continue
        if start:
            if line.strip().split("=")[0].strip() == "journal":
                journal = get_inb(line)
                if titlecase(journal) in journals:
                    abbrev = abbreviations[titlecase(journal)]
                    new_lines += replace_inb(line,abbrev)
                else:
                    new_lines += [line]
            elif line.strip().split("=")[0].strip() == "title":
                title = get_inb(line)
                new_title = titlecase(title)
                new_lines += replace_inb(line,new_title)
            elif line.strip().split("=")[0].strip() == "note":
                if "Publisher:" in line:
                    continue
                if "_eprint" in line:
                    continue
                if line[-2] != ",": #Remove multi-line notes? lol
                    startskip = True
                    continue
                else:
                    new_lines += [line]
            elif startskip:
                if line[-2] != ",":
                    continue
                else:
                    startskip = False
                    continue
            else:
                new_lines += [line]
        else:
            new_lines += [line]

    with open(bibfile_name.split(".bib")[0] + "_2.bib","w+") as file:
        for line in new_lines:
            file.write(line)

