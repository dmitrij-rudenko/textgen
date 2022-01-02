import * as fs from 'fs';
import * as tf from '@tensorflow/tfjs';
import { TextData } from './data';
import {generateText} from './model';

async function main() {
  const args = {}

  if (args.gpu) {
    console.log('Using GPU');
    require('@tensorflow/tfjs-node-gpu');
  } else {
    console.log('Using CPU');
    require('@tensorflow/tfjs-node');
  }

  // Load the model.
  const model = await tf.loadLayersModel('file://model/model.json');
  const sampleLen = 200
  const fullText = fs.readFileSync('./data.txt', { encoding: 'utf-8' });
  const fullTextData = new TextData('text-data', fullText, sampleLen, 60);
  const text = ' сразу после университета я встречалась с этим Верноном Дурслем'
  const textData = new TextData('text-data', text, sampleLen, 60);

  const ind = textData.textToIndices(text)
  console.log(ind, 'ind')
  
  const [seed, seedIndices] = textData.getRandomSlice();
  
  console.log(`Seed text:\n"${seed}"\n`);

  const generated = await generateText(model, fullTextData, ind, 60, 0.75);
  
  console.log(`Generated text:\n"${generated}"\n`);

}

(main)()