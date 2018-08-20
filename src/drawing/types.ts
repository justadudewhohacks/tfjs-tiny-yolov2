export type DrawBoxOptions = {
  lineWidth?: number
  color?: string
}

export type DrawTextOptions = {
  lineWidth?: number
  fontSize?: number
  fontStyle?: string
  color?: string
}

export type DrawDetectionOptions = {
  lineWidth?: number
  fontSize?: number
  fontStyle?: string
  textColor?: string
  boxColor?: string,
  withScore?: boolean,
  withClassName?: boolean
}

export type DrawOptions = {
  lineWidth: number
  fontSize: number
  fontStyle: string
  textColor: string
  boxColor: string,
  withScore: boolean,
  withClassName: boolean
}