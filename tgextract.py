# thx yehor https://raw.githubusercontent.com/egorsmkv/tg-extract-history/refs/heads/master/extractor.py
import os
import logging
import csv

from click import command, option

from telethon import TelegramClient, events, sync
from telethon.tl.types import Channel, PeerChannel
from telethon.tl.patched import Message

os.chdir('exp/tg')

def set_last_offset(group, value):
    with open(f'{group}.offset', 'w') as handle:
        handle.write(value)


def get_last_offset(group):
    if not os.path.exists(f'{group}.offset'):
        return 0
    with open(f'{group}.offset', 'r') as handle:
        val = handle.read()
        if not val:
            return 0
        return int(val)


@command()
@option('--debug', type=bool, default=False, help='Enable logging')
@option('--group', help='Group for extraction', required=True)
@option('--offset-id', type=int, default=0, help='Offset message ID')
@option('--save-offset', type=bool, default=False, help='Enable saving offset value into a file')
@option('--reset-downloading', type=bool, default=True, help='It takes an offset value from the file and append new messages into specified file.')
@option('--api-id', required=True)
@option('--api-hash', required=True)
@option('--session', type=str, default='extraction', help='makes sqlite files')
def run(debug, group, offset_id, save_offset, reset_downloading, api_id, api_hash, session):
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)

    client = TelegramClient(session, api_id, api_hash)
    client.start()

    try:
        raw = client.get_entity(group)
        if not isinstance(raw, Channel):
            exit('It is not a group')
    except ValueError as e:
        exit(e)

    channel = client.get_entity(PeerChannel(raw.id))

    mode = 'w'
    if reset_downloading:
        mode = 'a'
        offset_value = get_last_offset(group)
    else:
        offset_value = offset_id

    with open(f'{group}.csv', mode=mode) as handle:
        writer = csv.writer(handle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        if not reset_downloading:
            writer.writerow(['User ID', 'Message ID', 'Date', 'Message'])

        for item in client.iter_messages(channel.id, offset_id=offset_value):
            # ignore messages that are not directly users' messages
            if not isinstance(item, Message):
                continue

            strdate = item.date.strftime('%Y-%m-%d %H:%M:%S.%f %Z%z')
            writer.writerow([item.from_id, item.id, strdate, item.message])

            if reset_downloading:
                set_last_offset(group, str(item.id))


if __name__ == '__main__':
    run()
