import type { ComponentProps } from 'react'
import { Link } from 'react-router-dom'
import { navigateService, type RouterName, type RouteParams } from '@/services/navigate.service'

type LinkToProps = Omit<ComponentProps<typeof Link>, 'to'> & {
  name: RouterName
  params?: RouteParams
}

/** Resolves a named route alias + params to a path and renders a router Link. */
export function LinkTo({ name, params, ...rest }: LinkToProps) {
  return <Link to={navigateService.getPath(name, params)} {...rest} />
}
