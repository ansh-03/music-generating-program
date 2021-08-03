import numpy as np
from scipy.io.wavfile import write
import cv2
from skimage.metrics import structural_similarity as compare_ssim
import statistics
 
def piano_notes():   #music octave
    b_f = 261.63 #Frequency of Note C4
    notes = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B']
    freq = [b_f*(2**(a/12)) for a in range(len(notes))]
    octave = dict(zip(notes, freq))
    octave[''] = 0
    return octave
 
samplerate = 44100 #Frequecy in Hz
 
def get_wave(freq, duration=0.5): #produce a wave
    amplitude = 4096
    t = np.linspace(0, duration, int(samplerate * duration))
    wave = amplitude * np.sin(2 * np.pi * freq * t)
    return wave
 
def get_song_data(music_notes): #add all the waves to make the song
    note_freqs = piano_notes() # Function that we made earlier
    song = [get_wave(note_freqs[n]) for n in music_notes]
    song = np.concatenate(song)
    return song
 
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)
 
def extract_images(cam):
    pics = []
    curr, next = 1, 1
    while(True):  
        ret, frame = cam.read() #frames from videos
        if ret and curr == next:
            pics.append(auto_canny(cv2.cvtColor(cv2.resize(frame, (500, 600), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY)))
            next+=15
        elif curr != next: 
            curr+=1
            # prev = frame
        else:
            break
    # pics.append(auto_canny(cv2.cvtColor(cv2.resize(prev, (500, 600), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY)))
    return pics
 
def corr(data):
    #correlating no. of different points and musical notes
    image_median = statistics.median(data)
    freq_median = statistics.median([261.63*(2**(a/12)) for a in range(12)])
    return image_median / freq_median
 
def difference_btw_images(i1, i2):
    (score, diff) = compare_ssim(i1, i2, full=True)
    diff = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    b, number = np.unique(thresh, return_counts= True)
    return number[1]
 
def image_to_sound(image_data):
    music_notes = []
 
    for d in list(image_data):
        if d >= 491:
            music_notes.append('B')
        elif d >= 466:
            music_notes.append('a')
        elif d >= 440:
            music_notes.append('A')
        elif d >= 415:
            music_notes.append('g')
        elif d >= 392:
            music_notes.append('G')
        elif d >= 370:
            music_notes.append('f')
        elif d >= 349:
            music_notes.append('F')
        elif d >= 329:
            music_notes.append('E')
        elif d >= 311:
            music_notes.append('d')
        elif d >= 293:
            music_notes.append('D')
        elif d >= 277:
            music_notes.append('c')
        elif d >= 261:
            music_notes.append('C')
        else:
            music_notes.append('')
 
    data = get_song_data(music_notes)
    data = data * (16300/np.max(data)) # Adjusting the Amplitude
    return data
 
if __name__ == "__main__":
    try:
        file_name = input()
        cam = cv2.VideoCapture(file_name)
    except:
        print('File doesn\'t exist.')
        exit()
 
    images = extract_images(cam)
    
    difference = list()
    for i in range(len(images)-1):
        difference.append(difference_btw_images(images[i], images[i+1]))
    
    factor = corr(difference)
    difference = np.array(difference)
    sound = image_to_sound(difference//factor)
    write(‘%s.wav’%file_name, samplerate, sound.astype(np.int16))
    cam.release() 
    cv2.destroyAllWindows()