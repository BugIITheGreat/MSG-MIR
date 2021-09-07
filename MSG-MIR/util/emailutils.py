import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import os

username = "hljcfriend@163.com"
password = "NOXJIMWEXNWYKGOV"


def emailSend(msg):
    try:
        smtp = smtplib.SMTP()
        smtp.connect('smtp.163.com', 25)
        smtp.login(username, password)
        smtp.sendmail(username, username, msg.as_string())
        smtp.quit()
    except smtplib.SMTPConnectError as e:
        print('邮件发送失败，连接失败:', e.smtp_code, e.smtp_error)
    except smtplib.SMTPAuthenticationError as e:
        print('邮件发送失败，认证错误:', e.smtp_code, e.smtp_error)
    except smtplib.SMTPSenderRefused as e:
        print('邮件发送失败，发件人被拒绝:', e.smtp_code, e.smtp_error)
    except smtplib.SMTPRecipientsRefused as e:
        print('邮件发送失败，收件人被拒绝:', e.smtp_code, e.smtp_error)
    except smtplib.SMTPDataError as e:
        print('邮件发送失败，数据接收拒绝:', e.smtp_code, e.smtp_error)
    except smtplib.SMTPException as e:
        print('邮件发送失败, ', e.message)
    except Exception as e:
        print('邮件发送异常, ', str(e))


def emailAtta(imgPath, log_name, epoch):
    msg = MIMEMultipart('mixed')

    msg['Subject'] = "Training Data SEN"
    msg['From'] = "hljcfriend@163.com"
    msg['To'] = "hljcfriend@163.com"

    msgtext = MIMEText("epoch " + str(epoch) + "training data.", "plain", "utf-8")
    msg.attach(msgtext)

    with open(log_name, 'r') as f:
        attachfile = MIMEApplication(f.read())
    attachfile.add_header('Content-Disposition', 'attachment', filename=log_name)
    msg.attach(attachfile)

    listi = os.listdir(imgPath)
    listi.sort()
    for i, image in enumerate(listi[-6:]):
        with open(os.path.join(imgPath, image), 'rb') as fp:
            msgImage = MIMEImage(fp.read())
        msgImage.add_header('Content-ID', 'imgid'+str(i), filename=image)
        msg.attach(msgImage)

    return msg


'''
if __name__ == '__main__':
    msg = emailAtta('/home/hanlin/github/pytorch-CycleGAN-and-pix2pix/imgs/edges2cats.jpg')
    emailSend(msg)
'''
