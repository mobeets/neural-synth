var osc, env;
var noise, noiseEnv;
var part;
var part2;
var curX, curY;

var ellipseSize = 30;

var bpm = 60;

var drumSound1;
var drumSound2;
var drumPart1 = [0, 0, 1, 0];
var drumPart2 = [1, 1, 1, 1];
var snarePart = [1, 1, 1, 1];
var drumPhrases = [];

var bassPart1 = [60, 60, 57, 57];
var bassPart2 = [67, 67, 69, 69];
var bassParts = [bassPart1, bassPart2];
var bassInsts = [];
var bassPhrases = [];

var vae;
var nUserInsts = 10;
var userInsts = [];

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
      $("#model-output").html(data.output.join(", "));
      updateUserInsts(data.output);
    }
  });
}

function preload(){
  soundFormats('ogg', 'mp3');
  drumSound1 = loadSound('static/audio/drum');
  drumSound2 = loadSound('static/audio/studio-b');
}

function setup() {

  vae = new vaeModel();

  var windowSize = min(displayWidth, displayHeight/2);
  var canvas = createCanvas(windowSize, windowSize);
  // var canvas = createCanvas(displayWidth, displayHeight/2);
  // var canvas = createCanvas(width, height);
  canvas.parent("canvas-container");
  canvas.mousePressed(mousePressedOnCanvas);
  curX = displayHeight/2;
  curY = displayWidth/2;

  for (var i=0; i<nUserInsts; i++) {
    userInsts[i] = new getOsc(i+10);    
  }

  // init snare
  noise = new p5.Noise('brown');
  noise.start();
  noiseEnv = new p5.Env(0.01, 0.5, 0.1, 0);
  noiseEnv.setInput(noise);

  // init drum phrases
  part = new p5.Part(snarePart.length, 1/8);
  drumPhrases[0] = new p5.Phrase('snare', playSnare, snarePart);
  drumPhrases[1] = new p5.Phrase('drum1', playDrum1, drumPart1);
  drumPhrases[2] = new p5.Phrase('drum2', playDrum2, drumPart2);

  // init bass phrases
  for (var i=0; i<bassParts.length; i++) {
    bassInsts[i] = new getOsc(i);
    var playNote = bassInsts[i].playNote.bind(bassInsts[i]);
    bassPhrases[i] = new p5.Phrase('bass' + i.toString(), playNote, bassParts[i]);
  }

  part.setBPM(bpm);
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

function playSnare(time, params) {
  noiseEnv.play(noise, time);
}

function playDrum1(time, params) {
  drumSound1.play();
}
function playDrum2(time, params) {
  drumSound2.play();
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

function mousePressedOnCanvas() {
  var margin = ellipseSize/2 + 1;
  curX = constrain(mouseX, margin, width-margin);
  curY = constrain(mouseY, margin, height-margin);
  var Z1 = map(curY, 0, height, -3, 3);
  var Z2 = map(curX, 0, width, -3, 3);
  vae.decodeFromPosition(Z1, Z2);
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
    part.removePhrase('snare');
    part.removePhrase('drum1');
    part.removePhrase('drum2');
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
