//var spacyNLP = require('spacy-nlp');
var spacyNLP = require('../index.js');
var serverPromise = spacyNLP.server({ port: process.env.IOPORT, debug: true })
