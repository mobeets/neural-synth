var osc, env;
var noise, noiseEnv;
var part;
var part2;
var curX, curY;

var ellipseSize = 30;

var bpm = 70;

var drumSoundSnare;
var drumSoundHihat;
var drumSoundBassDrum;
var drumPartSnare = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1];
var drumPartHihat = [1, 0];
var drumPartBassDrum = [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0];
var drumPhrases = [];
var reverb1;
var reverb2;
var reverb3;

var bassInsts = [];
var bassPhrases = [];

var vae;
var nUserInsts = 10;
var userInsts = [];

function makeBassPart(theme, nPerNote, offsets, offsetReps) {
  var bassPart = [];
  var c = 0;
  for (var i=0; i<offsets.length; i++) {
    var nPerMeasure = offsetReps[i];
    for (var j=0; j<nPerMeasure; j++) {
      for (var k=0; k<theme.length; k++) {
        for (var l=0; l<nPerNote; l++) {
          bassPart[c] = theme[k] + offsets[i];
          c += 1;
        }
      }
    }
  }
  return bassPart;
}

var bassPart1 = makeBassPart([60, 60, 60, 60], 2, [-12, -12], [3, 1]);
var bassPart2 = makeBassPart([67, 67, 69, 69], 2, [-12, -12], [3, 1]);
var bassParts = [bassPart1, bassPart2];

function vaeModel() {
  this.position = [0, 0];
  this.output = new Float32Array(27 * 27);
  this.postUrl = "/decode";
}

vaeModel.prototype.decodeFromPosition = function(x,y) {
  this.position = [x,y];
  this.runModel(this.position);
}

vaeModel.prototype.runModel = function(position) {
  $.ajax({
    type: "POST",
    url: this.postUrl,
    data: JSON.stringify(position),
    contentType: 'application/json',
    dataType: 'json',
    error: function(data) {
      console.log("ERROR");
      console.log(data);
    },
    success: function(data) {
      updateUserInsts(data.output);
      detectChord(data.output);
    }
  });
}

function detectChord(notes) {
  $.ajax({
    type: "POST",
    url: "/detect",
    data: JSON.stringify(notes),
    contentType: 'application/json',
    dataType: 'json',
    error: function(data) {
      console.log("ERROR");
      console.log(data);
    },
    success: function(data) {
      $("#model-output").html(data.output);
    }
  });
}


function preload(){
  soundFormats('ogg', 'mp3');
  drumSoundSnare = loadSound('static/audio/snare');
  drumSoundHihat = loadSound('static/audio/hihat');
  drumSoundBassDrum = loadSound('static/audio/bass-drum');

  reverb1 = new p5.Reverb();
  reverb2 = new p5.Reverb();
  reverb3 = new p5.Reverb();
  reverb1.process(drumSoundHihat, 2, 5);
  reverb2.process(drumSoundSnare, 2, 5);
  reverb3.process(drumSoundBassDrum, 2, 5);
}

function setup() {

  // init VAE
  vae = new vaeModel();

  // init window
  var windowSize = min(displayWidth, displayHeight/2);
  var canvas = createCanvas(windowSize, windowSize);
  canvas.parent("canvas-container");
  canvas.mousePressed(mousePressedOnCanvas);
  curX = displayHeight/2;
  curY = displayWidth/2;

  // init user synth
  for (var i=0; i<nUserInsts; i++) {
    userInsts[i] = new getUserOsc(i+10);    
  }

  // init drum phrases
  part = new p5.Part(bassParts[0].length, 1/16);
  drumPhrases[0] = new p5.Phrase('drum-snare', playDrum1, drumPartSnare);
  drumPhrases[1] = new p5.Phrase('drum-hihat', playDrum2, drumPartHihat);
  drumPhrases[2] = new p5.Phrase('drum-bass', playBassDrum, drumPartBassDrum);

  // init bass phrases
  for (var i=0; i<bassParts.length; i++) {
    bassInsts[i] = new getBassOsc(i);
    var playNote = bassInsts[i].playNote.bind(bassInsts[i]);
    bassPhrases[i] = new p5.Phrase('bass' + i.toString(), playNote, bassParts[i]);
  }

  // start playing part
  part.setBPM(bpm);
  part.loop();
  part.start();

}

function getBassOsc(ind) {
  this.ind = ind;
  this.osc = new p5.SinOsc();
  this.osc.amp(0.0);
  this.osc.start();
  // this.envelope = new p5.Env(0.015, 0.16, 0.20, 0.12, 3.6, 0.00);
  this.envelope = new p5.Env(0.01, 0.5, 0.20, 0.4, 0.0, 0.0);
}

getBassOsc.prototype.playNote = function(time, params) {
  this.note = params;
  this.freqValue = midiToFreq(this.note);
  this.osc.freq(this.freqValue);
  this.envelope.play(this.osc, 0, 0.2);
}

function getUserOsc(ind) {
  this.ind = ind;
  this.osc = new p5.SinOsc();
  this.osc.amp(0.0);
  this.envelope = new p5.Env(0.01, 0.16, 0.40, 0.1, 3.6, 0.0);
  this.osc.start();
}

getUserOsc.prototype.playNote = function(time, params) {
  this.note = params;
  this.freqValue = midiToFreq(this.note);
  this.osc.freq(this.freqValue);
  this.envelope.play(this.osc);
}

function playDrum1(time, params) {
  drumSoundSnare.setVolume(0.4);
  drumSoundSnare.play();
}
function playDrum2(time, params) {
  drumSoundHihat.setVolume(0.4);
  drumSoundHihat.play();
}
function playBassDrum(time, params) {
  drumSoundBassDrum.setVolume(0.4);
  drumSoundBassDrum.play();
}

function updateUserInsts(notes) {
  for (var i=0; i<nUserInsts; i++) {
    if (i > notes.length) {
      userInsts[i].playNote(0, 0);
    } else {  
      userInsts[i].playNote(0, notes[i]);
    }
  }
}

var curMilli;
var lastMilli;
function mousePressedOnCanvas() {
  var margin = ellipseSize/2 + 1;
  curX = constrain(mouseX, margin, width-margin);
  curY = constrain(mouseY, margin, height-margin);
  var Z1 = map(curY, 0, height, -3, 3);
  var Z2 = map(curX, 0, width, -3, 3);
  vae.decodeFromPosition(Z1, Z2);
  return false;
}

// draw a ball mapped to current latent position
function draw() {
  background(245);
  // fill(0);
  // strokeWeight(0);
  ellipse(curX, curY, ellipseSize, ellipseSize);
}

function toggleDrums() {
  if($(this).hasClass('active')) {
    // part.stop();
    part.removePhrase('drum-snare');
    part.removePhrase('drum-hihat');
    part.removePhrase('drum-bass');
  } else {
    // part.start();
    part.addPhrase(drumPhrases[0]);
    part.addPhrase(drumPhrases[1]);
    part.addPhrase(drumPhrases[2]);
  }
}
function toggleBass() {
  if($(this).hasClass('active')) {
    for (var i=0; i<bassParts.length; i++) {
      part.removePhrase('bass' + i.toString());
    }
  } else {
    for (var i=0; i<bassParts.length; i++) {
      part.addPhrase(bassPhrases[i]);
    }
  }
}

function addHandlers() {
  $("#drums-toggle").click(toggleDrums);
  $("#bass-toggle").click(toggleBass);
}

$( document ).ready(function() {
  addHandlers();
});
