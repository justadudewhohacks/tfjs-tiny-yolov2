function getImageUri(imageName) {
  return `images/${imageName}`
}

async function fetchImage(uri) {
  return (await fetch(uri)).blob()
}

async function fetchJson(uri) {
  return (await fetch(uri)).json()
}


async function requestExternalImage(imageUrl) {
  const res = await fetch('fetch_external_image', {
    method: 'post',
    headers: {
      'content-type': 'application/json'
    },
    body: JSON.stringify({ imageUrl })
  })
  if (!(res.status < 400)) {
    console.error(res.status + ' : ' + await res.text())
    throw new Error('failed to fetch image from url: ' + imageUrl)
  }

  let blob
  try {
    blob = await res.blob()
    return await faceapi.bufferToImage(blob)
  } catch (e) {
    console.error('received blob:', blob)
    console.error('error:', e)
    throw new Error('failed to load image from url: ' + imageUrl)
  }
}

function renderSelectList(selectListId, onChange, initialValue, renderChildren) {
  const select = document.createElement('select')
  $(selectListId).get(0).appendChild(select)
  renderChildren(select)
  $(select).val(initialValue)
  $(select).on('change', (e) => onChange(e.target.value))
  $(select).material_select()
}

function renderOption(parent, text, value) {
  const option = document.createElement('option')
  option.innerHTML = text
  option.value = value
  parent.appendChild(option)
}

function renderImageSelectList(selectListId, onChange, initialValue) {
  const images = [1, 2, 3, 4, 5].map(idx => `bbt${idx}.jpg`)
  function renderChildren(select) {
    images.forEach(imageName =>
      renderOption(
        select,
        imageName,
        getImageUri(imageName)
      )
    )
  }

  renderSelectList(
    selectListId,
    onChange,
    getImageUri(initialValue),
    renderChildren
  )
}