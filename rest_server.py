from aiohttp import web, multipart
import aiohttp_cors
import asyncio
import os
from ocr import ocr, ocrlight
from formatter import merge

routes = web.RouteTableDef()


@routes.post('/recognizer')
async def handler(request):
    reader = await request.multipart()
    tmpid = None
    filename = None
    while True:
        field = await reader.next()
        if field is None:
            break
        if field.name == "file":
            filename = field.filename
            size = 0
            with open(os.path.join('pdf', filename), 'wb') as f:
                while True:
                    chunk = await field.read_chunk()
                    if not chunk:
                        break
                    size += len(chunk)
                    f.write(chunk)
        if field.name == "tmpid":
            tmpid = await field.text()
            print(tmpid)

    result = ocrlight("pdf/"+str(filename))
    # result = merge(result)
    result["properties"]["fileName"]["value"] = filename
    return  web.json_response(result)

if __name__ == "__main__":

    app = web.Application()
    app.add_routes(routes)

    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })
    for route in list(app.router.routes()):
        cors.add(route)

    runner = web.AppRunner(app)
    asyncio.get_event_loop().run_until_complete(runner.setup())
    site = web.TCPSite(runner, host='0.0.0.0', port=6000)

    loop = asyncio.get_event_loop()

    try:
        asyncio.ensure_future(site.start())
        print("start")
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        loop.call_soon_threadsafe(loop.stop)  # here
        print('Finished!')
        loop.close()