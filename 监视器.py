'''
@Time : 2023-12-09 17:06
@Author : laolao
@FileName: 监视器.py
'''

import smtplib
import time
from email.mime.text import MIMEText
from jupyter_client import KernelManager

def send_email():
    # 邮件内容
    msg = MIMEText('你的代码已经运行结束！', 'plain', 'utf-8')
    msg['From'] = 'coraisname@foxmail.com'
    msg['To'] = 'ydjhlsz@qq.com'
    msg['Subject'] = '代码运行结束'

    # 发送邮件
    smtp_server = 'smtp.qq.com'
    smtp_port = 465
    smtp_user = 'coraisname@foxmail.com'
    smtp_password = 'xuxtgdldkothebbc'
    smtp = smtplib.SMTP_SSL(smtp_server, smtp_port)
    smtp.login(smtp_user, smtp_password)
    smtp.sendmail(smtp_user, [msg['To']], msg.as_string())
    smtp.quit()

def main():
    # 连接到jupyter服务器
    km = KernelManager()
    km.start_kernel()
    print(km)
    kc = km.client()
    print(kc)
    kc.start_channels()

    # 检查内核是否仍在运行
    while kc.is_alive():
        time.sleep(1)

    # 发送邮件通知
    send_email()

if __name__ == '__main__':
    import torch.nn as nn
    nn.LogSigmoid

    main()
