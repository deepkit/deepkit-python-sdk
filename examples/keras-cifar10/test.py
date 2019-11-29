import asyncio

import deepkit

context = deepkit.context('deepkit.yml')

acc = deepkit.create_metric('acc')


async def bla():
    for i in range(1000):
        await asyncio.sleep(1)
        context.epoch(i, 10)
        acc.send(i, i * 2)
        print("asdasd")


asyncio.get_event_loop().run_until_complete(bla())
