"""Helper functions for using telegram."""
import telegram
import json

# %%
# infofile = 'info.json'


# %%
def make_telegram_bot(infofile):
    """Make a telegram bot from info json file with token and chat id.

    Returns the bot and the chat id.
    """
    with open(infofile, "r") as f:
        info = json.loads(f.read())

    bot = telegram.Bot(info["token"])

    return bot, info["id"]


# # %%
# bot, id = make_telegram_bot(infofile)
# bot.send_message(id, "test message 1 2 3")


# %%
def send_gif(gif, bot, id, timeout=100):
    """Send a gif using the bot to the given chat id.

    Actually sends a mp4 video.
    """
    with open(gif, "rb") as f:
        bot.send_video(id, f, timeout=timeout)


# %%
# send_gif("gifs/neural_ode_gifs/neural_ode_1.mp4", bot, id)
