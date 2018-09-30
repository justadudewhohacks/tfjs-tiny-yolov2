require('./voc/.env')

const express = require('express')
const path = require('path')
const fs = require('fs')

const app = express()

app.use(express.json())
app.use(express.urlencoded({ extended: true }))

const public = path.join(__dirname, './voc')
app.use(express.static(public))
app.use(express.static(path.join(__dirname, './js')))
app.use(express.static(path.join(__dirname, '../examples/public')))
app.use(express.static(path.join(__dirname, '../models')))
app.use(express.static(path.join(__dirname, '../dist')))
app.use(express.static(path.join(__dirname, './node_modules/file-saver')))

app.get('/', (req, res) => res.redirect('/train'))
app.get('/train', (req, res) => res.sendFile(path.join(public, 'trainVoc.html')))
app.get('/verify', (req, res) => res.sendFile(path.join(public, 'verifyVoc.html')))

const trainDataPath = path.resolve(process.env.TRAIN_DATA_PATH)
const imagesPath = path.join(trainDataPath, 'images')
const groundTruthPath = path.join(trainDataPath, 'ground_truth')
app.use(express.static(imagesPath))
app.use(express.static(groundTruthPath))

const groundTruthFilenames = fs.readdirSync(groundTruthPath)

app.get('/voc_train_filenames', (req, res) => res.status(202).send(groundTruthFilenames))

app.listen(8000, () => console.log('Listening on port 8000!'))