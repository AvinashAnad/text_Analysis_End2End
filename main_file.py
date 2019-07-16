import extraction as ext
import ca
import ppt
import email

searchstring = str(input("Enter the Search search\n>>>"))
datelimit = str(input("search for current date - (Enter Days)"))
ext.extractfn(searchstring,datelimit)
ca.cleanNanalyse(searchstring)
ppt.ppt()
email.sendemail()
