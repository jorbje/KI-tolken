import edge_tts
import asyncio

async def main():
    communicate = edge_tts.Communicate("Dette er en meget formel test af det danske sprog. Vi håber dette fungerer fint, så vi kan se transskriptionen.", "da-DK-ChristelNeural")
    await communicate.save("c:/Dev/Tolk/test_da.mp3")

asyncio.run(main())
