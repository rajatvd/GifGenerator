# Generate gifs using neural ODEs and send to a telegram chat

Randomly initialized neural ODEs can be used to generate neat looking animations by solving the ODE until a large end time. This repo contains a script which periodically generates gifs using this method, and sends them to a telegram chat using a telegram bot. Here's an example gif:


<p align="center">
  <img width=448px height=448px src="example.gif">
</p>

Include your [telegram bot](https://core.telegram.org/bots) details in a file called `info.json` with two keys, a `token` and a chat `id`.


Run the service as:

`python gif_generation_service.py with <config_updates>`

Add your own method of generating gifs to the `gif_generators.py` module using the `export` decorator. It must return the filename of the generated gif (actually an `mp4` file, as currently only those can be viewed inline in the chat when sent as a video.)