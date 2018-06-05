# Download the helper library from https://www.twilio.com/docs/python/install
# pip install https://www.twilio.com/docs/sms/quickstart/python
from twilio.rest import Client


def send_sms(msg):
    # Your Account Sid and Auth Token from twilio.com/console
    account_sid = ''
    auth_token = ''
    client = Client(account_sid, auth_token)

    message = client.messages.create(
                                body=msg,
                                from_='+11111111111',
                                to='+10000000000'
                            )

    print(message.sid)


if __name__ == '__main__':
    send_sms("Loss:{:.3f}  F_Car: {:.3f}  F_Road: {:.3f} F_: {:.3f}".format(0.01,0.53,0.55,0.5558)) 