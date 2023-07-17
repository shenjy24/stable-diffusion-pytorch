import uuid
import datetime


def generate_random_str():
    random_uuid = uuid.uuid4()
    filename = str(random_uuid).replace("-", "")
    return filename


def get_current_time():
    current_datetime = datetime.datetime.now()
    return current_datetime


if __name__ == '__main__':
    random = generate_random_str()
    print(random)
    print(get_current_time())
