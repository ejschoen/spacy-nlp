// assuming server is started and ready
const polyIO = require('poly-socketio')
/* istanbul ignore next */
process.env.IOPORT = process.env.IOPORT || 6466

// call the python spacy nlp parser via socketIO
// output: [{text, len, tokens, noun_phrases, parse_tree, parse_list}]
/* istanbul ignore next */
function parse(text,segment) {
  polyIO.client({ port: process.env.IOPORT })
  var msg = {
      input: [text,segment],
      to: 'nlp.cgkb-py',
      intent: 'parse'
  }
  return global.client.pass(msg)
    .then((reply) => {
      return reply.output
    })
}

// call the python spacy nlp parser via socketIO
// output: [{text, len, tokens, noun_phrases, parse_tree, parse_list}]
/* istanbul ignore next */
function split(text) {
  polyIO.client({ port: process.env.IOPORT })
  var msg = {
      input: text,
      to: 'nlp.cgkb-py',
      intent: 'split'
  }
  return global.client.pass(msg)
    .then((reply) => {
      return reply.output
    })
}

// parse('Bob Brought the pizza to Alice.')
//   .then((output) => {
//     console.log(output)
//     console.log(JSON.stringify(output[0].parse_tree, null, 2))
//       // console.log(JSON.stringify(output[0].parse_list, null, 2))
//   })

function train_ner(data,types) {
    polyIO.client({ port: process.env.IOPORT });
    var msg = {
	input: [data, types],
	to: 'nlp.cgkb-py',
	intent: 'train_ner'
    }
    return global.client.pass(msg)
	.then ((reply) => {
	    return reply.output
	});
}


module.exports = {
    parse: parse,
    train_ner: train_ner
}
