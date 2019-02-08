"""Generate and send gifs at time intervals."""
from functools import partial

from sacred import Experiment
from apscheduler.schedulers.blocking import BlockingScheduler

from gif_generators import GIFFERS, CONFIGS
from telegram_helpers import make_telegram_bot, send_gif

ex = Experiment("gif_generation")


@ex.config
def main_config():
    """Make main config."""
    giffer = 'neural_ode'
    ex.add_config({giffer: CONFIGS[giffer]})

    infofile = 'info.json'

    hours = 1
    minutes = 0
    seconds = 0

    count = 2  # how many gifs to generate each run


@ex.capture
def make_and_send_gif(giffer, bot, id, _config, _log):
    """Make gif using the giffer with the config.

    Send it using the bot to the given id.
    """
    gif = GIFFERS[giffer](**_config[giffer], _log=_log)
    send_gif(gif, bot, id)


def job(infofile, count):
    """Job which makes and sends count number of gifs.

    Uses telegram bot from infofile.
    """
    bot, id = make_telegram_bot(infofile)
    for i in range(count):
        make_and_send_gif(bot=bot, id=id)


@ex.automain
def main(giffer, infofile,
         hours, minutes, seconds, count,
         _config, _log):

    job(infofile, count)

    sched = BlockingScheduler()
    sched.add_job(partial(job, infofile, count), 'interval',
                  hours=hours, minutes=minutes, seconds=seconds)
    sched.start()
