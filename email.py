def sendemail():
    import smtplib,ssl
    from email.mime.multipart import MIMEMultipart
    from email.mime.base import MIMEBase
    from email.mime.text import MIMEText
    from email.utils import formatdate
    from email import encoders
    import passwd as pswd
    pwd=pswd.passwdrret()
    msg = MIMEMultipart()
    msg['From'] = '<from_email>'
    msg['To'] = '<to_email>'
    msg['Date'] = formatdate(localtime = True)
    msg['Subject'] = 'sending test.pptx - take1'
    msg.attach(MIMEText('<EmailBody>'))

    part = MIMEBase('application', "octet-stream")
    part.set_payload(open("Presentation.pptx", "rb").read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', 'attachment; filename="test.pptx"')
    msg.attach(part)

    #context = ssl.SSLContext(ssl.PROTOCOL_SSLv3)
    #SSL connection only working on Python 3+
    smtp = smtplib.SMTP('smtp.gmail.com', 587  )
    smtp.starttls()
    smtp.login('<gmail username>',pwd)
    smtp.sendmail('<from_email>', '<to_email>', msg.as_string())
    smtp.quit()