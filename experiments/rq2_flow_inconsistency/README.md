# Setup Android in a Linux Container

This assumes that the server is using a GPU (NVIDIA) that is not compatible with Wayland and requires the use of Weston compositor to launch Waydroid.

1. Install and Start Waydroid

```
https://docs.waydro.id/usage/install-on-desktops
```

2. Configure Waydroid

Assuming that we need a portrait orientation
```
waydroid prop set persist.waydroid.width 480
waydroid prop set persist.waydroid.height 854
```

2. Install and Launch Weston

```
sudo apt install weston
weston
```

3. Launching Waydroid Session

Inside Weston, open a new terminal and type:
```
waydroid show-full-ui
```
