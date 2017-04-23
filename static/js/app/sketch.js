var osc, env;
var noise, noiseEnv;
var part;
var curX, curY;

var bpm = 60;
var snarePart = [0, 0, 1, 0];

var bassPart1 = [50, 55, 55, 55];
var bassPart2 = [60, 60, 62, 62];
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

vaeModel.prototype.selectCoordinates = function(x,y) {
  this.position = [x,y];
  console.log(this.position);
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

function setup() {

  vae = new vaeModel();

  createCanvas(500, 500);
  curX = mouseX;
  curY = mouseY;

  // prepare oscillator controlled by user
  // oscUser = new p5.SinOsc();
  // oscUser.start();
  // oscUser.amp(1.0);

  for (var i=0; i<nUserInsts; i++) {
    userInsts[i] = new getOsc(i+10);    
  }

  // init snare
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

  // build part
  part = new p5.Part(bassParts[0].length, 1/4);
  // part.addPhrase('snare', playSnare, snarePart);
  // for (var i=0; i<bassParts.length; i++) {
  //   part.addPhrase(bassPhrases[i]);
  // }
  // part.setBPM(bpm);
  // part.loop();
  // part.start();

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

function updateUserInsts(notes) {
  for (var i=0; i<nUserInsts; i++) {
    if (i > notes.length) {
      userInsts[i].playNote(0, 0);
    } else {  
      userInsts[i].playNote(0, notes[i]);
    }
  }
}

function mousePressed() {
  curX = mouseX;
  curY = mouseY;
  var Z1 = map(curY, 0, height, -3, 3);
  var Z2 = map(curX, 0, width, -3, 3);
  vae.selectCoordinates(Z1, Z2);
}

// draw a ball mapped to current latent position
function draw() {
  background(200);
  ellipse(curX, curY, 30, 30);
}
