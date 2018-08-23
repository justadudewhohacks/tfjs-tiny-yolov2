import { Rect } from './Rect';
import { Dimensions } from './types';

export class ObjectDetection {
  private _score: number
  private _classScore: number
  private _className: string
  private _box: Rect
  private _imageWidth: number
  private _imageHeight: number

  constructor(
    score: number,
    classScore: number,
    className: string,
    relativeBox: Rect,
    imageDims: Dimensions
  ) {
    const { width, height } = imageDims
    this._imageWidth = width
    this._imageHeight = height
    this._score = score
    this._classScore = classScore
    this._className = className
    this._box = new Rect(
      relativeBox.x * width,
      relativeBox.y * height,
      relativeBox.width * width,
      relativeBox.height * height
    )
  }

  public get score(): number {
    return this._score
  }

  public get classScore(): number {
    return this._classScore
  }

  public get className(): string {
    return this._className
  }

  public get box(): Rect {
    return this._box
  }

  public get imageWidth(): number {
    return this._imageWidth
  }

  public get imageHeight(): number {
    return this._imageHeight
  }

  public get relativeBox(): Rect {
    return new Rect(
      this._box.x / this._imageWidth,
      this._box.y / this._imageHeight,
      this._box.width / this._imageWidth,
      this._box.height / this._imageHeight
    )
  }

  public getScore() {
    return this.score
  }

  public getBox() {
    return this.box
  }

  public getImageWidth() {
    return this.imageWidth
  }

  public getImageHeight() {
    return this.imageHeight
  }

  public getRelativeBox() {
    return this.relativeBox
  }

  public forSize(width: number, height: number): ObjectDetection {
    return new ObjectDetection(
      this.score,
      this.classScore,
      this.className,
      this.getRelativeBox(),
      { width, height}
    )
  }
}