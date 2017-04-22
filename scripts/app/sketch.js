/**
 *  Create a sequence using a Part.
 *  Add two Phrases to the part, and tell the part to loop.
 *
 *  The callback includes parameters (the value at that position in the Phrase array)
 *  as well as time, which should be used to schedule playback with precision.
 */

var osc, env; // used by playNote
var noise, noiseEnv; // used by playSnare
var part; // a part we will loop
var curX, curY;

var snarePart = [0, 0, 1, 0];

var bassPart1 = [50, 55, 55, 55];
var bassPart2 = [60, 60, 62, 62];
var bassParts = [bassPart1, bassPart2];
var bassInsts = [];
var bassPhrases = [];

function setup() {
  createCanvas(500, 500);

  curX = mouseX;
  curY = mouseY;

  // prepare synth controlled by user
  oscUser = new p5.SinOsc();
  oscUser.start();
  oscUser.amp(0);

  // prepare the noise and env used by playSnare()
  noise = new p5.Noise();
  noise.start();
  noiseEnv = new p5.Env(0.01, 0.5, 0.1, 0);
  noiseEnv.setInput(noise);

  // init bass parts
  for (var i=0; i<bassParts.length; i++) {
    bassInsts[i] = new getOsc(i);
    var playNote = bassInsts[i].playNote.bind(bassInsts[i]);
    bassPhrases[i] = new p5.Phrase('bass' + i.toString(), playNote, bassParts[i]);
  }

  // create a part with 8 spaces
  part = new p5.Part(bassParts[0].length, 1/4);
  // part.addPhrase('snare', playSnare, snarePart);
  for (var i=0; i<bassParts.length; i++) {
    part.addPhrase(bassPhrases[i]);
  }
  part.setBPM(60);
  part.loop();
  part.start();

}

function getOsc(ind) {
  this.ind = ind;
  this.osc = new p5.SinOsc();
  this.osc.amp(0.0);
  this.envelope = new p5.Env(0.015, 0.16, 0.20, 0.12, 3.6, 0.00);
  this.osc.start();
}

getOsc.prototype.playNote = function(time, params) {
  this.note = params;
  this.freqValue = midiToFreq(this.note);
  this.osc.freq(this.freqValue);
  this.envelope.play(this.osc);
}

function playBass(time, params) {
  osc.freq(midiToFreq(params), 0, time);
  env.play(osc, time);
}

function playSnare(time, params) {
  noiseEnv.play(noise, time);
}

function mousePressed() {
  // set freq
  var freq = map(mouseX, 0, width, 40, 880);
  oscUser.freq(freq);

  // set pitch
  var volume = map(mouseY, 0.1, height, 0, 2);
  oscUser.amp(volume);

  curX = mouseX;
  curY = mouseY;

}

// draw a ball mapped to current note height
function draw() {
  background(200);
  ellipse(curX, curY, 30, 30);
}
