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
  const text = fs.readFileSync('./data.txt', { encoding: 'utf-8' });
  const textData = new TextData('text-data', text, sampleLen, 60);

  const [seed, seedIndices] = textData.getRandomSlice();
  
  console.log(`Seed text:\n"${seed}"\n`);

  const generated = await generateText(
    model, textData, seedIndices, 60, 1);
  
  console.log(`Generated text:\n"${generated}"\n`);

}

(main)()