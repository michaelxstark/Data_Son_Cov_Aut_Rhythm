import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import sounddevice as sd

# reading in data
df = pd.read_csv('CovidFaelle_Timeline.csv', sep=';')

# displaying first ten, checking columns
df.head(10)
lof = list(df.columns)
lof

# cleaning data

# changing to proper floats
df['SiebenTageInzidenzFaelle'] = df['SiebenTageInzidenzFaelle'].str.replace(',', '.')
df['SiebenTageInzidenzFaelle'].head(20)

# getting rid of every tenth value (whole of austria)
df = df.loc[df['BundeslandID'] % 10 != 0, :].reset_index()
df.tail(10)


# scaling values for volume, size and color


BIDS = list(set(df['BundeslandID']))
AmpSize = [np.array(df.loc[(df['BundeslandID'] == bid), 'AnzahlFaelle'])
           for bid in BIDS]


Color = [np.array(df.loc[(df['BundeslandID'] == bid),
                         'SiebenTageInzidenzFaelle']) for bid in BIDS]


Pitch = [np.array(df.loc[(df['BundeslandID'] == bid),
                         'SiebenTageInzidenzFaelle']) for bid in BIDS]

Rep = [np.array(df.loc[(df['BundeslandID'] == bid),
                       'SiebenTageInzidenzFaelle']) for bid in BIDS]

# Converter function to floats


def strtofloat(a):
    for i in range(len(a)):
        a[i] = float(a[i])
    return a


Color2 = [np.apply_along_axis(strtofloat, 0, Color[j])
          for j in range(len(Color))]

# scale funtion for 'colorchanges' and pitchchanges
Pitch2 = [np.apply_along_axis(strtofloat, 0, Pitch[j])
          for j in range(len(Pitch))]
Pitch2

Rep2 = [np.apply_along_axis(strtofloat, 0, Rep[j])
        for j in range(len(Rep))]


def pv(ar):
    for i in range(len(ar)):
        if ar[i] < 50:
            ar[i] = 1
        elif 50 < ar[i] < 100:
            ar[i] = 1.5
        elif 100 < ar[i] < 150:
            ar[i] = 2
        elif 150 < ar[i] < 250:
            ar[i] = 2.25
        elif ar[i] > 250:
            ar[i] = 3
    return ar


Pitches = [np.apply_along_axis(pv, 0, Pitch2[j])
           for j in range(len(Pitch2))]


def cc(ar):
    for i in range(len(ar)):
        if ar[i] < 50:
            ar[i] = 'lightgreen'
        elif 50 < ar[i] < 100:
            ar[i] = 'green'
        elif 100 < ar[i] < 150:
            ar[i] = 'yellow'
        elif 150 < ar[i] < 250:
            ar[i] = 'orange'
        elif ar[i] > 250:
            ar[i] = 'red'
    return ar


Color3 = [np.apply_along_axis(cc, 0, Color2[j])
          for j in range(len(Color2))]


# last preparations

AmpSize1 = []
for i in range(len(AmpSize[0])):
    for j in range(len(AmpSize)):
        AmpSize1.append(AmpSize[j][i])
AmpSize1

Color4 = []
for x in range(len(Color3[0])):
    for y in range(len(Color3)):
        Color4.append(Color3[y][x])


# AudioEngine

sd.query_devices()
sd.default.device = 'BlackHole 16ch, Core Audio'

# simple sine-osc with ad-env


def sine(frq, a, d):
    sr = 44100
    env = np.concatenate((np.linspace(0, 0.5, int(round(sr * a, 0))),
                          np.linspace(0.5, 0, int(round(sr * d, 0)))))
    t = np.arange(int(round(d * sr, 0)) + int(round(a * sr, 0))) / sr
    sine = 1 * np.sin(2 * np.pi * frq * t) * env
    return sine


# simple panning - algorithm
def panner(x, angle):
    # pan a mono audio source into stereo
    # x is a numpy array, angle is the angle in radiants
    left = np.sqrt(2)/2.0 * (np.cos(angle) - np.sin(angle)) * x
    right = np.sqrt(2)/2.0 * (np.cos(angle) + np.sin(angle)) * x
    return np.dstack((left, right))[0]


# Scaling to values between 0 and 1
Amps = [np.array(df.loc[(df['BundeslandID'] == bid), 'AnzahlFaelle'])
        for bid in BIDS]
Amps2 = [np.array(Amps[i] / Amps[i].max()) for i in range(len(Amps))]
Amps2

# setting samplerate
sr = 44100
# notelength
dur = 0.4
# attack and decay of tones
a = dur * 0.01
d = dur * 0.99

# assign basefreqeuncies to counties based on their latitude

latitude_df = pd.read_html('https://www.distancelatlong.com/country/austria')
latitude_df[2].loc[:, ['States', 'Latitude']]
basefreqs = [110, 110 * 1.5, 220, 440 * (3/4),
             440 * (9/8), 550, 440 * (15/8), 880, 990]


lat_dict = {latitude_df[2].loc[i, 'States']:
            latitude_df[2].loc[i, 'Latitude']
            for i in range(len(latitude_df[2]))}


lat_dict_sort = sorted(lat_dict.items(), key=lambda x: x[1])


zipped_lat_freq = list(zip(lat_dict_sort, basefreqs))


basefreqs_lat = [sorted(zipped_lat_freq)[i][1]
                 for i in range(len(zipped_lat_freq))]


# defining variations in pitch

Pitches2 = [Pitches[i] * basefreqs_lat[i] for i in range(len(Pitches))]
Pitches2

# defining repetitions


def reps(ar):
    for i in range(len(ar)):
        if ar[i] < 50:
            ar[i] = 1
        elif 50 < ar[i] < 100:
            ar[i] = 2
        elif 100 < ar[i] < 150:
            ar[i] = 3
        elif 150 < ar[i] < 250:
            ar[i] = 4
        elif ar[i] > 250:
            ar[i] = 6
    return ar


Reps = [np.apply_along_axis(reps, 0, Rep2[j])
        for j in range(len(Rep2))]

Reps

# making tuples for reps and pitchchanges


Pitches2
Amps2
p_r = [[] for i in range(len(Reps))]
for i in range(len(Reps)):
    for j in range(len(Reps[i])):
        p_r[i].append((Reps[i][j], Pitches2[i][j]))
p_r2 = [np.array(i) for i in p_r]
p_r[0]

sine_pat = [[] for i in range(len(p_r))]
sine_pat
for i in range(len(sine_pat)):
    for j in p_r[i]:
        sine_pat[i].append(np.tile(sine(j[1], a / int(j[0]),
                                        d / int(j[0])), int(j[0])))

# useless. anyways.
sine_pat2 = [i for i in sine_pat]

sine_pat3 = [[] for j in range(len(sine_pat2))]
for i in range(len(sine_pat3)):
    for j in range(len(Amps2[i])):
        sine_pat3[i].append(sine_pat2[i][j] * Amps2[i][j])


# plot

plt.style.available
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(20, 8))
ax.set_xticks(df['BundeslandID'][:9])
ax.set_xticklabels(list(df['Bundesland'][:9]))
ax.set_ylim(-2, 12)
ax.set_frame_on(False)
ax.axes.get_yaxis().set_visible(True)
ax.axes.get_xaxis().set_visible(True)
ax.set_yticklabels([])
ax.grid(False, axis='both')
ax.set_title('Covid19_Cases_in_Austria')

x = np.array(list(set(df['BundeslandID'])))

# scaling and applying latitude values
scaled_lat = list(zip(lat_dict_sort, list(range(1, 10))))
scaled_lat2 = sorted(scaled_lat)
scaled_lat3 = [i[1] for i in scaled_lat2]

y = scaled_lat3


lines = ax.scatter(x, y,
                   marker='o',
                   s=50,
                   c='green', alpha=0.8)

plt.close()


def animate(i):
    lines.set_sizes(np.array(AmpSize1[i:i+9]) * 5)
    lines.set_color(Color4[i:i+9])
    ax.set_ylabel(df['Time'][i][:10])
    if i == len(df) - 9:
        sd.play((panner(np.concatenate(sine_pat3[0]), np.radians(-40)) +
                 panner(np.concatenate(sine_pat3[1]), np.radians(-30)) +
                 panner(np.concatenate(sine_pat3[2]), np.radians(-20)) +
                 panner(np.concatenate(sine_pat3[3]), np.radians(-10)) +
                 panner(np.concatenate(sine_pat3[4]), np.radians(0)) +
                 panner(np.concatenate(sine_pat3[5]), np.radians(10)) +
                 panner(np.concatenate(sine_pat3[6]), np.radians(20)) +
                 panner(np.concatenate(sine_pat3[7]), np.radians(30)) +
                 panner(np.concatenate(sine_pat3[8]), np.radians(40))) * 0.5,
                sr)
    return lines,


animation = FuncAnimation(fig, func=animate,
                          frames=np.arange(9, len(df), 9),
                          interval=dur * 1000,
                          blit=False, repeat=False)


HTML(animation.to_html5_video())
