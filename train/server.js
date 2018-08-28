require('./.env')

const express = require('express')
const path = require('path')
const fs = require('fs')
const { get } = require('request')

const app = express()

app.use(express.json())
app.use(express.urlencoded({ extended: true }))

const public = path.join(__dirname, 'public')
app.use(express.static(public))
app.use(express.static(path.join(__dirname, '../examples/public')))
app.use(express.static(path.join(__dirname, '../models')))
app.use(express.static(path.join(__dirname, '../dist')))
app.use(express.static(path.join(__dirname, './node_modules/file-saver')))

app.get('/', (req, res) => res.redirect('/train_voc'))
app.get('/train_voc', (req, res) => res.sendFile(path.join(public, 'trainVoc.html')))
app.get('/verify_voc', (req, res) => res.sendFile(path.join(public, 'verifyVoc.html')))

const trainDataPath = path.resolve(process.env.TRAIN_DATA_PATH)
const imagesPath = path.join(trainDataPath, 'images')
const groundTruthPath = path.join(trainDataPath, 'ground_truth')
app.use(express.static(imagesPath))
app.use(express.static(groundTruthPath))

const groundTruthFilenames = fs.readdirSync(groundTruthPath)

app.get('/voc_train_filenames', (req, res) => res.status(202).send(groundTruthFilenames))


app.post('/fetch_external_image', async (req, res) => {
  const { imageUrl } = req.body
  if (!imageUrl) {
    return res.status(400).send('imageUrl param required')
  }
  try {
    const externalResponse = await request(imageUrl)
    res.set('content-type', externalResponse.headers['content-type'])
    return res.status(202).send(Buffer.from(externalResponse.body))
  } catch (err) {
    return res.status(404).send(err.toString())
  }
})

app.listen(8000, () => console.log('Listening on port 8000!'))

function request(url, returnBuffer = true, timeout = 10000) {
  return new Promise(function(resolve, reject) {
    const options = Object.assign(
      {},
      {
        url,
        isBuffer: true,
        timeout,
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36'
        }
      },
      returnBuffer ? { encoding: null } : {}
    )

    get(options, function(err, res) {
      if (err) return reject(err)
      return resolve(res)
    })
  })
}