require('./face/.env')

const express = require('express')
const path = require('path')
const fs = require('fs')

const app = express()

app.use(express.json())
app.use(express.urlencoded({ extended: true }))

const public = path.join(__dirname, 'face')
app.use(express.static(public))
app.use(express.static(path.join(__dirname, './js')))
app.use(express.static(path.join(__dirname, '../examples/public')))
app.use(express.static(path.join(__dirname, '../models')))
app.use(express.static(path.join(__dirname, '../dist')))
app.use(express.static(path.join(__dirname, './node_modules/file-saver')))

app.get('/', (req, res) => res.redirect('/train'))
app.get('/train', (req, res) => res.sendFile(path.join(public, 'train.html')))
app.get('/verify', (req, res) => res.sendFile(path.join(public, 'verify.html')))
app.get('/test', (req, res) => res.sendFile(path.join(public, 'test.html')))

const trainDataPath = path.resolve(process.env.TRAIN_DATA_PATH)
const testDataPath = path.resolve(process.env.TEST_DATA_PATH)
const imagesPath = path.join(trainDataPath, 'jpgs')
const groundTruthPath = path.join(trainDataPath, 'ground_truth')
app.use(express.static(imagesPath))
app.use(express.static(groundTruthPath))
app.use(express.static(testDataPath))

const trainIds = fs.readdirSync(groundTruthPath).map(gt => gt.replace('.json', ''))
const testIds = JSON.parse(fs.readFileSync(path.join(testDataPath, 'test_ground_truth.json')).toString())
  .map(gt => gt.imgFile.replace('.jpg', ''))

app.get('/train_ids', (req, res) => res.status(202).send(trainIds))
app.get('/test_ids', (req, res) => res.status(202).send(testIds))

app.listen(8000, () => console.log('Listening on port 8000!'))